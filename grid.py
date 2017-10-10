import argparse, os, random, shutil
import numpy as np, scipy.sparse
import cPickle as pickle
import tables

from utils import dijkstra
from utils.dotdict import dotdict
from utils.qmdp import QMDP

try:
    import ipdb as pdb
except Exception:
    import pdb


FREESTATE = 0.0
OBSTACLE = 1.0


class GridBase(object):
    def __init__(self, params):
        """
        Initialize domain simulator
        :param params: domain descriptor dotdict
        :param db: pytable database file
        """
        self.params = params

        self.N = params.grid_n
        self.M = params.grid_m
        self.grid_shape = [self.N, self.M]
        self.moves = params.moves
        self.observe_directions = params.observe_directions

        self.num_action = params.num_action
        self.num_obs = params.num_obs
        self.obs_len = len(self.observe_directions)
        self.num_state = self.N * self.M

        self.grid = None

    def simulate_policy(self, policy, grid, b0, start_state, goal_states, first_action=None):
        params = self.params
        max_traj_len = params.traj_limit

        if first_action is None:
            first_action = params.stayaction

        self.grid = grid

        self.gen_pomdp()
        qmdp = self.get_qmdp(goal_states)

        state = start_state
        reward_sum = 0.0  # accumulated reward
        gamma_acc = 1.0

        collisions = 0
        failed = False
        step_i = 0

        # initialize policy
        env_img = grid[None]
        goal_img = self.process_goals(goal_states)
        b0_img = self.process_beliefs(b0)
        policy.reset(env_img, goal_img, b0_img)

        while True:
            # finish if state is terminal, i.e. we reached a goal state
            if all([np.isclose(qmdp.T[x][state, state], 1.0) for x in range(params.num_action)]):
                assert state in goal_states
                break

            # stop if trajectory limit reached
            if step_i >= max_traj_len:  # it should reach terminal state sooner or later
                failed = True
                break

            # choose next action
            if step_i == 0:
                act = first_action
            else:
                act = policy.eval(act, self.obs_lin_to_bin(obs))

            # simulate action
            state, r = qmdp.transition(state, act)
            obs = qmdp.random_obs(state, act)

            reward_sum += r * gamma_acc
            gamma_acc = gamma_acc * qmdp.discount

            # count collisions
            if np.isclose(r, params.R_obst):
                collisions += 1

            step_i += 1

        traj_len = step_i

        return (not failed), traj_len, collisions, reward_sum

    def generate_trajectories(self, db, num_traj):
        params = self.params
        max_traj_len = params.traj_limit

        for traj_i in range(num_traj):
            # generate a QMDP object, initial belief, initial state and goal state
            # also generates a random grid for the first iteration
            qmdp, b0, start_state, goal_states = self.random_instance(generate_grid=(traj_i == 0))

            qmdp.solve()

            state = start_state
            b = b0.copy()  # linear belief
            reward_sum = 0.0  # accumulated reward
            gamma_acc = 1.0

            beliefs = [] # includes start and goal
            states = [] # includes start and goal
            actions = [] # first action is always stay. Excludes action after reaching goal
            observs = [] # Includes observation at start but excludes observation after reaching goal

            collisions = 0
            failed = False
            step_i = 0

            while True:
                beliefs.append(b)
                states.append(state)

                # finish if state is terminal, i.e. we reached a goal state
                if all([np.isclose(qmdp.T[x][state, state], 1.0) for x in range(params.num_action)]):
                    assert state in goal_states
                    break

                # stop if trajectory limit reached
                if step_i >= max_traj_len:  # it should reach terminal state sooner or later
                    failed = True
                    break

                # choose action
                if step_i == 0:
                    # dummy first action
                    act = params.stayaction
                else:
                    act = qmdp.qmdp_action(b)

                # simulate action
                state, r = qmdp.transition(state, act)
                bprime, obs, b = qmdp.belief_update(b, act, state_after_transition=state)

                actions.append(act)
                observs.append(obs)

                reward_sum += r * gamma_acc
                gamma_acc = gamma_acc * qmdp.discount

                # count collisions
                if np.isclose(r, params.R_obst):
                    collisions += 1

                step_i += 1

            # add to database
            if not failed:
                db.root.valids.append([len(db.root.samples)])

            traj_len = step_i

            # step: state (linear), action, observation (linear)
            step = np.stack([states[:traj_len], actions[:traj_len], observs[:traj_len]], axis=1)

            # sample: env_id, goal_state, step_id, traj_length, collisions, failed
            # length includes both start and goal (so one step path is length 2)
            sample = np.array(
                [len(db.root.envs), goal_states[0], len(db.root.steps), traj_len, collisions, failed], 'i')

            db.root.samples.append(sample[None])
            db.root.bs.append(np.array(beliefs[:1]))
            db.root.expRs.append([reward_sum])
            db.root.steps.append(step)

        # add environment only after adding all trajectories
        db.root.envs.append(self.grid[None])

    def random_instance(self, generate_grid=True):
        """
        Generate a random problem instance for a grid.
        Picks a random initial belief, initial state and goal states.
        :param generate_grid: generate a new grid and pomdp model if True, otherwise use self.grid
        :return:
        """
        while True:
            if generate_grid:
                self.grid = self.random_grid(self.params.grid_n, self.params.grid_m, self.params.Pobst)
                self.gen_pomdp()  # generates pomdp model, self.T, self.Z, self.R

            while True:
                # sample initial belief, start, goal
                b0, start_state, goal_state = self.gen_start_and_goal()
                if b0 is None:
                    assert generate_grid
                    break  # regenerate obstacles

                goal_states = [goal_state]

                # reject if start == goal
                if start_state in goal_states:
                    continue

                # create qmdp
                qmdp = self.get_qmdp(goal_states)  # makes soft copies from self.T{R,Z}simple
                # it will also convert to csr sparse, and set qmdp.issparse=True

                return qmdp, b0, start_state, goal_states

    def gen_pomdp(self):
        # construct all POMDP model(R, T, Z)
        self.Z = self.build_Z()
        self.T, Tml, self.R = self.build_TR()

        # transform into graph with opposite directional actions, so we can compute path from goal
        G = {i: {} for i in range(self.num_state)}
        for a in range(self.num_action):
            for s in range(self.num_state):
                snext = Tml[s, a]
                if s != snext:
                    G[snext][s] = 1  # edge with distance 1
        self.graph = G

    def build_Z(self):
        params = self.params

        Pobs_succ = params.Pobs_succ

        Z = np.zeros([self.num_action, self.num_state, self.num_obs], 'f')

        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j])
                state = self.state_bin_to_lin(state_coord)

                # first build observation
                obs = np.zeros([self.obs_len])  # 1 or 0 in four directions
                for direction in range(self.obs_len):
                    neighb = self.apply_move(state_coord, np.array(self.observe_directions[direction]))
                    if self.check_free(neighb):
                        obs[direction] = 0
                    else:
                        obs[direction] = 1

                # add all observations with their probabilities
                for obs_i in range(self.num_obs):
                    dist = np.abs(self.obs_lin_to_bin(obs_i) - obs).sum()
                    prob = np.power(1.0 - Pobs_succ, dist) * np.power(Pobs_succ, self.obs_len - dist)
                    Z[:, state, obs_i] = prob

                # sanity check
                assert np.isclose(1.0, Z[0, state, :].sum())

        return Z

    def build_TR(self):
        """
        Builds transition (T) and reward (R) model for a grid.
        The model does not capture goal states, which must be incorporated later.
        :return: transition model T, maximum likely transitions Tml, reward model R
        """
        params = self.params
        Pmove_succ = params.Pmove_succ

        # T, R does not capture goal state, it must be incorporated later
        T = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
             for x in range(self.num_action)]  # probability of transition with a0 from s1 to s2
        R = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
             for x in range(self.num_action)]  # probability of transition with a0 from s1 to s2
        # goal will be defined as a terminal state, all actions remain in goal with 0 reward

        # maximum likely versions
        Tml = np.zeros([self.num_state, self.num_action], 'i')  # Tml[s, a] --> next state
        Rml = np.zeros([self.num_state, self.num_action], 'f')  # Rml[s, a] --> reward after executing a in s

        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j])
                state = self.state_bin_to_lin(state_coord)

                # build T and R
                for act in range(self.num_action):
                    neighbor_coord = self.apply_move(state_coord, np.array(self.moves[act]))
                    if self.check_free(neighbor_coord):
                        Rml[state, act] = params['R_step'][act]
                    else:
                        neighbor_coord[:2] = [i, j]  # dont move if obstacle or edge of world
                        # alternative: neighbor_coord = state_coord
                        Rml[state, act] = params['R_obst']

                    neighbor = self.state_bin_to_lin(neighbor_coord)
                    Tml[state, act] = neighbor
                    if state == neighbor:
                        # shortcut if didnt move
                        R[act][state, state] = Rml[state, act]
                        T[act][state, state] = 1.0
                    else:
                        R[act][state, state] = params['R_step'][act]
                        # cost if transition fails (might be lucky and avoid wall)
                        R[act][state, neighbor] = Rml[state, act]
                        T[act][state, state] = 1.0 - Pmove_succ
                        T[act][state, neighbor] = Pmove_succ

        return T, Tml, R

    def gen_start_and_goal(self, maxtrials=1000):
        """
        Pick an initial belief, initial state and goal state randomly
        """
        free_states = np.nonzero((self.grid == FREESTATE).flatten())[0]
        freespace_size = len(free_states)

        for trial in range(maxtrials):
            b0sizes = np.floor(freespace_size / np.power(2.0, np.arange(20)))
            b0sizes = b0sizes[:np.nonzero(b0sizes < 1)[0][0]]
            b0size = int(np.random.choice(b0sizes))

            b0ind = np.random.choice(free_states, b0size, replace=False)
            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size  # uniform distribution over sampled states

            # sanity check
            for state in b0ind:
                coord = self.state_lin_to_bin(state)
                assert self.check_free(coord)

            # sample initial state from initial belief
            start_state = np.random.choice(self.num_state, p=b0)

            # sample goal uniformly from free space
            goal_state = np.random.choice(free_states)

            # check if path exists from start to goal, if not, pick a new set
            D, path_pointers = dijkstra.Dijkstra(self.graph, goal_state)  # map of distances and predecessors
            if start_state in D:
                break
        else:
            # never succeeded
            raise ValueError

        return b0, start_state, goal_state

    def get_qmdp(self, goal_states):
        qmdp = QMDP(self.params)

        qmdp.processT(self.T)  # this will make a hard copy
        qmdp.processR(self.R)
        qmdp.processZ(self.Z)

        qmdp.set_terminals(goal_states, reward=self.params.R_goal)

        qmdp.transfer_all_sparse()
        return qmdp

    @staticmethod
    def sample_free_state(map):
        """
        Return the coordinates of a random free state from the 2D input map
        """
        while True:
            coord = [random.randrange(map.shape[0]), random.randrange(map.shape[1])]
            if map[coord[0],coord[1],0] == FREESTATE:
                return coord

    @staticmethod
    def outofbounds(map, coord):
        return (coord[0] < 0 or coord[0] >= map.shape[0] or coord[1] < 0 or coord[1] >= map.shape[1])

    @staticmethod
    def apply_move(coord_in, move):
        coord = coord_in.copy()
        coord[:2] += move[:2]
        return coord

    def check_free(self, coord):
        return (not GridBase.outofbounds(self.grid, coord) and self.grid[coord[0], coord[1]] != OBSTACLE)

    @staticmethod
    def random_grid(N, M, Pobst):
        grid = np.zeros([N, M])

        # borders
        grid[0, :] = OBSTACLE
        grid[-1, :] = OBSTACLE
        grid[:, 0] = OBSTACLE
        grid[:, -1] = OBSTACLE

        rand_field = np.random.rand(N, M)
        grid = np.array(np.logical_or(grid, (rand_field < Pobst)), 'i')
        return grid

    def obs_lin_to_bin(self, obs_lin):
        obs = np.array(np.unravel_index(obs_lin, [2,2,2,2]), 'i')
        if obs.ndim > 2:
            raise NotImplementedError
        elif obs.ndim > 1:
            obs = np.transpose(obs, [1,0])
        return obs

    def obs_bin_to_lin(self, obs_bin):
        return np.ravel_multi_index(obs_bin, [2,2,2,2])

    def state_lin_to_bin(self, state_lin):
        return np.unravel_index(state_lin, self.grid_shape)

    def state_bin_to_lin(self, state_coord):
        return np.ravel_multi_index(state_coord, self.grid_shape)

    @staticmethod
    def create_db(filename, params, total_env_count=None, traj_per_env=None):
        """
        :param filename: file name for database
        :param params: dotdict describing the domain
        :param total_env_count: total number of environments in the dataset (helps to preallocate space)
        :param traj_per_env: number of trajectories per environment
        """
        N = params.grid_n
        M = params.grid_m
        num_state = N * M
        if total_env_count is not None and traj_per_env is not None:
            total_traj_count = total_env_count * traj_per_env
        else:
            total_traj_count = 0

        if os.path.isfile(filename):
            print (filename + " already exitst, opening.")
            return tables.open_file(filename, mode='a')

        db = tables.open_file(filename, mode='w')

        db.create_earray(db.root, 'envs', tables.IntAtom(), shape=(0, N, M), expectedrows=total_env_count)

        db.create_earray(db.root, 'expRs', tables.FloatAtom(), shape=(0, ), expectedrows=total_traj_count)

        db.create_earray(db.root, 'valids', tables.IntAtom(), shape=(0, ), expectedrows=total_traj_count)

        db.create_earray(db.root, 'bs', tables.FloatAtom(), shape=(0, num_state), expectedrows=total_traj_count)

        db.create_earray(db.root, 'steps', tables.IntAtom(),
                         shape=(0, 3),  # state,  action, observation
                         expectedrows=total_traj_count * 10) # rough estimate

        db.create_earray(db.root, 'samples', tables.IntAtom(),
                         shape=(0, 6),  # env_id, goal_state, step_id, traj_length, collisions, failed
                         expectedrows=total_traj_count)
        return db

    def process_goals(self, goal_state):
        """
        :param goal_state: linear goal state
        :return: goal image, same size as grid
        """
        goal_img = np.zeros([goal_state.shape[0], self.N, self.M], 'i')
        goalidx = np.unravel_index(goal_state, [self.N, self.M])

        goal_img[np.arange(goal_state.shape[0]), goalidx[0], goalidx[1]] = 1

        return goal_img

    def process_beliefs(self, linear_belief):
        """
        :param linear_belief: belief in linear space
        :return: belief reshaped to grid size
        """
        batch = (linear_belief.shape[0] if linear_belief.ndim > 1 else 1)
        b = linear_belief.reshape([batch, self.params.grid_n, self.params.grid_m, ])
        if b.dtype != np.float:
            return b.astype('f')

        return b


def generate_grid_data(path, N=30, M=30, num_env=10000, traj_per_env=5, Pmove_succ=1.0, Pobs_succ=1.0):
    """
    :param path: path for data file. use separate folders for training and test data
    :param N: grid rows
    :param M: grid columnts
    :param num_env: number of environments in the dataset (grids)
    :param traj_per_env: number of trajectories per environment (different initial state, goal, initial belief)
    :param Pmove_succ: probability of transition succeeding, otherwise stays in place
    :param Pobs_succ: probability of correct observation, independent in each direction
    """

    params = dotdict({
        'grid_n': N,
        'grid_m': M,
        'Pobst': 0.25,  # probability of obstacles in random grid

        'R_obst': -10, 'R_goal': 20, 'R_step': -0.1,
        'discount': 0.99,
        'Pmove_succ':Pmove_succ,
        'Pobs_succ': Pobs_succ,

        'num_action': 5,
        'moves': [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]],  # right, down, left, up, stay
        'stayaction': 4,

        'num_obs': 16,
        'observe_directions': [[0, 1], [1, 0], [0, -1], [-1, 0]],
        })

    params['obs_len'] = len(params['observe_directions'])
    params['num_state'] = params['grid_n']*params['grid_m']
    params['traj_limit'] = 4 * (params['grid_n'] + params['grid_m'])
    params['R_step'] = [params['R_step']] * params['num_action']

    # save params
    if not os.path.isdir(path): os.mkdir(path)
    pickle.dump(dict(params), open(path + "/params.pickle", 'w'), -1)

    # randomize seeds, set to previous value to determinize random numbers
    np.random.seed()
    random.seed()

    # grid domain object
    domain = GridBase(params)

    # make database file
    db = GridBase.create_db(path+"data.hdf5", params, num_env, traj_per_env)

    for env_i in range(num_env):
        print ("Generating env %d with %d trajectories "%(env_i, traj_per_env))
        domain.generate_trajectories(db, num_traj=traj_per_env)

    print ("Done.")


def main():
    parser = argparse.ArgumentParser(description='Generate grid environments')
    parser.add_argument(
         'path', type=str,
         help='Directory for datasets')
    parser.add_argument(
         'train', type=int, default=10000,
         help='Number of training environments')
    parser.add_argument(
         'test', type=int, default=500,
         help='Number of test environments')
    parser.add_argument(
         '--N', type=int, default=10,
         help='Grid size')
    parser.add_argument(
         '--train_trajs', type=int, default=5,
         help='Number of trajectories per environment in the training set. 5 by default.')
    parser.add_argument(
         '--test_trajs', type=int, default=1,
         help='Number of trajectories per environment in the test set. 1 by default.')

    parser.add_argument(
         '--Pmove_succ', type=float, default=1.0,
         help='Probability of successful actions, 1.0 by default')
    parser.add_argument(
         '--Pobs_succ', type=float, default=1.0,
         help='Probability of successful observation (independently for each direction), 1.0 by default')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        answer = raw_input("%s exists. Do you want to remove it(y/n)?" % args.path)
        if answer != 'y':
            return
        shutil.rmtree(args.path)

    if not os.path.isdir(args.path): os.mkdir(args.path)

    # training data
    generate_grid_data(args.path + '/train/', N=args.N, M=args.N, num_env=args.train, traj_per_env=args.train_trajs,
                       Pmove_succ=args.Pmove_succ, Pobs_succ=args.Pobs_succ)

    # test data
    generate_grid_data(args.path + '/test/', N=args.N, M=args.N, num_env=args.test, traj_per_env=args.test_trajs,
                       Pmove_succ=args.Pmove_succ, Pobs_succ=args.Pobs_succ)


# default
if __name__ == "__main__":
    main()