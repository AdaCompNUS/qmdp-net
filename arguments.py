import pickle, argparse, os
import numpy as np
from utils.dotdict import dotdict


def parse_args(arglist):

    parser = argparse.ArgumentParser(description='Run training on gridworld')

    parser.add_argument('path',
                        help='Path to data folder containing train and test subfolders')
    parser.add_argument('--logpath', default='./log/',
                        help='Path to save log and trained model')

    parser.add_argument('--loadmodel', nargs='*',
                        help='Load model weights from checkpoint')

    parser.add_argument('--eval_samples', type=int,
                    default=100,
                    help='Number of samples to evaluate the learned policy on')
    parser.add_argument('--eval_repeats', type=int,
                    default=1,
                    help='Repeat simulating policy for a given number of times. Use 5 for stochastic domains')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of minibatches for training')
    parser.add_argument('--training_envs', type=float, default=0.9,
                        help='Proportion of training data used for trianing. Remainder will be used for validation')
    parser.add_argument('--step_size', type=int, default=4,
                        help='Number of maximum steps for backpropagation through time')
    parser.add_argument('--lim_traj_len', type=int, default=100,
                        help='Clip trajectories to a maximum length')
    parser.add_argument('--includefailed', action='store_true',
                        help='Include unsuccessful demonstrations in the training and validation set.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience_first', type=int,
                        default=30,
                        help='Start decaying learning rate if no improvement for a given number of steps')
    parser.add_argument('--patience_rest', type=int,
                        default=5,
                        help='Patience after decay started')
    parser.add_argument('--decaystep', type=int,
                        default=15,
                        help='Total number of learning rate decay steps')
    parser.add_argument('--epochs', type=int,
                        default=1000,
                        help='Maximum number of epochs')

    parser.add_argument('--cache', nargs='*',
                        default=['steps', 'envs', 'bs'],
                        help='Cache nodes from pytable dataset. Default: steps, envs, bs')

    parser.add_argument('-K', '--K', type=int,
                        default=-1,
                        help='Number of iterations of value iteration in QMDPNet. Compute from grid size if negative.')

    args = parser.parse_args(args=arglist)

    # load domain parameters
    params = dotdict(pickle.load(open(os.path.join(args.path, 'train/params.pickle'), 'rb')))

    # set default K
    if args.K < 0:
        args.K = 3 * params.grid_n

    # combine all parameters to a single dotdict
    for key in vars(args):
        params[key] = getattr(args, key)

    return params
