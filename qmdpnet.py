from tensorpack import graph_builder
import tensorflow as tf
import numpy as np


class QMDPNet:
    """
    Class implementing a QMDP-Net for the grid navigation domain
    """
    def __init__(self, params, batch_size=1, step_size=1):
        """
        :param params: dotdict describing the domain and network hyperparameters
        :param batch_size: minibatch size for training. Use batch_size=1 for evaluation
        :param step_size: limit the number of steps for backpropagation through time. Use step_size=1 for evaluation.
        """
        self.params = params
        self.batch_size = batch_size
        self.step_size = step_size

        self.placeholders = None
        self.context_tensors = None
        self.belief = None
        self.update_belief_op = None
        self.logits = None
        self.loss = None

    def build_placeholders(self):
        """
        Creates placeholders for all inputs in self.placeholders
        """
        N = self.params.grid_n
        M = self.params.grid_m
        obs_len = self.params.obs_len
        step_size = self.step_size
        batch_size = self.batch_size

        placeholders = []
        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(batch_size, N, M), name='In_map'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(batch_size, N, M), name='In_goal'))

        placeholders.append(tf.placeholder(tf.float32,
                                            shape=(batch_size, N, M),
                                            name='In_b0'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(batch_size,), name='In_isstart'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(step_size, batch_size), name='In_actions'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(step_size, batch_size, obs_len), name='In_local_obs'))

        placeholders.append(tf.placeholder(tf.float32,
                                            shape=(step_size, batch_size), name='In_weights'))

        placeholders.append(tf.placeholder(tf.uint8,
                                            shape=(step_size, batch_size), name='Label_a'))

        self.placeholders = placeholders

    def build_inference(self, reuse=False):
        """
        Creates placeholders, ops for inference and loss
        Unfolds filter and planner through time
        Also creates an op to update the belief. It should be always evaluated together with the loss.
        :param reuse: reuse variables if True
        :return: None
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        self.build_placeholders()

        map, goal, b0, isstart, act_in, obs_in, weight, act_label = self.placeholders # TODO clean up

        # types conversions
        map = tf.to_float(map)
        goal = tf.to_float(goal)
        isstart = tf.to_float(isstart)
        isstart = tf.reshape(isstart, [self.batch_size] + [1]*(b0.get_shape().ndims-1))
        act_in = tf.to_int32(act_in)
        obs_in = tf.to_float(obs_in)
        act_label = tf.to_int32(act_label)

        outputs = []

        # pre-compute context, fixed through time
        with tf.variable_scope("planner"):
            Q, _, _ = PlannerNet.VI(map, goal, self.params)
        with tf.variable_scope("filter"):
            Z = FilterNet.f_Z(map, self.params)

        self.context_tensors = [Q, Z]

        # create variable for hidden belief (equivalent to the hidden state of an RNN)
        self.belief = tf.Variable(np.zeros(b0.get_shape().as_list(), 'f'), trainable=False, name="hidden_belief")

        # figure out current b. b = b0 if isstart else blast
        b = (b0 * isstart) + (self.belief * (1-isstart))

        for step in range(self.step_size):
            # filter
            with tf.variable_scope("filter") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                b = FilterNet.beliefupdate(Z, b, act_in[step], obs_in[step], self.params)

            # planner
            with tf.variable_scope("planner") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                action_pred = PlannerNet.policy(Q, b, self.params)
                outputs.append(action_pred)

        # create op that updates the belief
        self.update_belief_op = self.belief.assign(b)

        # compute loss (cross-entropy)
        logits = tf.stack(values=outputs, axis=0)  # shape is [step_size, batch_size, num_action]

        # logits = tf.reshape(logits, [self.step_size*self.batch_size, self.params.num_action])
        # act_label = tf.reshape(act_label, [-1])
        # weight = tf.reshape(weight, [-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=act_label)

        # weight loss. weights are 0.0 for steps after the end of a trajectory, otherwise 1.0
        loss = loss * weight
        loss = tf.reduce_mean(loss, axis=[0, 1], name='xentropy')

        self.logits = logits
        self.loss = loss

    def build_train(self, initial_lr):
        """
        """
        #count_number_trainable_params(verbose=True) # TODO remove

        # Decay learning rate by manually incrementing decay_step
        decay_step = tf.Variable(0.0, name='decay_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            initial_lr, decay_step, 1, 0.8, staircase=True, name="learning_rate")

        trainable_variables = tf.trainable_variables()

        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9)
        # clip gradients
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0, use_norm=tf.global_norm(grads))

        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.train_op = train_op


class QMDPNetPolicy():
    """
    Policy wrapper for QMDPNet. Implements two functions: reset and eval.
    """
    def __init__(self, network, sess):
        self.network = network
        self.sess = sess

        self.belief_img = None
        self.env_img = None
        self.goal_img = None

        assert self.network.batch_size == 1 and self.network.step_size == 1

    def reset(self, env_img, goal_img, belief_img):
        #TODO
        """

        :param env_img:
        :param goal_img:
        :param belief_img:
        :return:
        """
        N = self.network.params.grid_n
        M = self.network.params.grid_m

        self.belief_img = belief_img.reshape([1, N, M])
        self.env_img = env_img.reshape([1, N, M])
        self.goal_img = goal_img.reshape([1, N, M])

        self.sess.run(tf.assign(self.network.belief, self.belief_img))
        #
        # feed_dict = {tf.get_default_graph().get_tensor_by_name('In_map:0'): env_img,
        #              tf.get_default_graph().get_tensor_by_name('In_goal:0'): goal_img}
        # self.context_value = self.sess.run(self.network.context_variables, feed_dict=feed_dict)

    def eval(self, last_act, last_obs):
        #TODO
        """

        :param last_act:
        :param last_obs:
        :return:
        """
        isstart = np.array([0])
        last_act = np.reshape(last_act, [1, 1])
        last_obs = np.reshape(last_obs, [1, 1, self.network.params.obs_len])

        # input data. do not neet weight and label for prediction
        data = [self.env_img, self.goal_img, self.belief_img, isstart, last_act, last_obs]
        feed_dict = {self.network.placeholders[i]: data[i] for i in range(len(self.network.placeholders)-2)}

        # evaluate QMDPNet
        logits, _ = self.sess.run([self.network.logits, self.network.update_belief_op], feed_dict=feed_dict)
        act = np.argmax(logits.flatten())

        return act


class PlannerNet():
    @staticmethod
    def f_R(map, goal, num_action):
        theta = tf.stack([map, goal], axis=3)
        R = conv_layers(theta, np.array([[3, 150, 'relu'], [1, num_action, 'lin']]), "R_conv")
        return R

    @staticmethod
    def VI(map, goal, params):
        """
        builds neural network implementing value iteration. this is the first part of planner module. Fixed through time.
        inputs: map (batch x N x N) and goal(batch)
        returns: Q_K, and optionally: R, list of Q_i
        """
        # build reward model R
        R = PlannerNet.f_R(map, goal, params.num_action)

        # get transition model Tprime. It represents the transition model in the filter, but the weights are not shared.
        kernel = FilterNet.f_T(params.num_action)

        # initialize value image
        V = tf.zeros(map.get_shape().as_list() + [1])
        Q = None

        # repeat value iteration K times
        for i in range(params.K):
            # apply transition and sum
            Q = tf.nn.conv2d(V, kernel, [1, 1, 1, 1], padding='SAME')
            Q = Q + R
            V = tf.reduce_max(Q, axis=[3], keep_dims=True)

        return Q, V, R

    @staticmethod
    def f_pi(q, num_action):
        action_pred = fc_layers(q, np.array([[num_action, 'lin']]), names="pi_fc")
        return action_pred

    @staticmethod
    def policy(Q, b, params, reuse=False):
        """
        second part of planner module
        :param Q: input Q_K after value iteration
        :param b: belief at current step
        :param params: params
        :return: a_pred,  vector with num_action elements, each has the
        """
        # weight Q by the belief
        b_tiled = tf.tile(tf.expand_dims(b, 3), [1, 1, 1, params.num_action])
        q = tf.multiply(Q, b_tiled)
        # sum over states
        q = tf.reduce_sum(q, [1, 2], keep_dims=False)

        # low-level policy, f_pi
        action_pred = PlannerNet.f_pi(q, params.num_action)

        return action_pred


class FilterNet():
    @staticmethod
    def f_Z(map, params, reuse=False):
        """
        This implements f_Z, outputs an observation model (Z). Fixed through time.
        inputs: map (NxN array)
        returns: Z
        """
        # CNN: theta -> Z
        map = tf.expand_dims(map, -1)
        Z = conv_layers(map, np.array([[3, 150, 'lin'], [1, 17, 'sig']]), "Z_conv")

        # normalize over observations
        Z_sum = tf.reduce_sum(Z, [3], keep_dims=True)
        Z = tf.div(Z, Z_sum + 1e-8)  # add a small number to avoid division by zero

        return Z

    @staticmethod
    def f_A(action, num_action):
        # identity function
        w_A = tf.one_hot(action, num_action)
        return w_A

    @staticmethod
    def f_O(local_obs):

        w_O = fc_layers(local_obs, np.array([[17, 'tanh'], [17, 'smax']]), names="O_fc")
        return w_O

    @staticmethod
    def f_T(num_action):
        # get transition kernel
        initializer = tf.truncated_normal_initializer(mean=1.0/9.0, stddev=1.0/90.0, dtype=tf.float32)
        kernel = tf.get_variable("w_T_conv", [3 * 3, num_action], initializer=initializer, dtype=tf.float32)

        # enforce proper probability distribution (i.e. values must sum to one) by softmax
        kernel = tf.nn.softmax(kernel, dim=0)
        kernel = tf.reshape(kernel, [3, 3, 1, num_action], name="T_w")

        return kernel

    @staticmethod
    def beliefupdate(Z, b, action, local_obs, params, reuse=False):
        """
        Belief update in the filter module with pre-computed Z.
        :param b: belief (b_i), [batch, N, M, 1]
        :param action: action input (a_i)
        :param obs: observation input (o_i)
        :return: updated belief b_(i+1)
        """
        # step 1: update belief with transition
        # get transition kernel (T)
        kernel = FilterNet.f_T(params.num_action)

        # apply convolution which corresponds to the transition function in an MDP (f_T)
        b = tf.expand_dims(b, -1)
        b_prime = tf.nn.conv2d(b, kernel, [1, 1, 1, 1], padding='SAME')

        # index into the appropriate channel of b_prime
        w_A = FilterNet.f_A(action, params.num_action)
        w_A = w_A[:, None, None]
        b_prime_a = tf.reduce_sum(tf.multiply(b_prime, w_A), [3], keep_dims=False) # soft indexing

        #b_prime_a = tf.abs(b_prime_a) # TODO there was this line. does it make a difference with softmax?

        # step 2: update belief with observation
        # get observation probabilities for the obseravtion input by soft indexing
        w_O = FilterNet.f_O(local_obs)
        w_O = w_O[:,None,None] #tf.expand_dims(tf.expand_dims(w_O, axis=1), axis=1)
        Z_o = tf.reduce_sum(tf.multiply(Z, w_O), [3], keep_dims=False) # soft indexing

        b_next = tf.multiply(b_prime_a, Z_o)

        # step 3: normalize over the state space
        # add small number to avoid division by zero
        b_next = tf.div(b_next, tf.reduce_sum(b_next, [1, 2], keep_dims=True) + 1e-8)

        return b_next


# Helper function to construct layers conveniently

def conv_layer(input, kernel_size, filters, name, w_mean=0.0, w_std=None, addbias=True, strides=(1, 1, 1, 1), padding='SAME'):
    """
    Create variables and operator for a convolutional layer
    :param input: input tensor
    :param kernel_size: size of kernel
    :param filters: number of convolutional filters
    :param name: variable name for convolutional kernel and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights. Use 1/sqrt(input_param_count) if None.
    :param addbias: add bias if True
    :param strides: convolutional strides, match TF
    :param padding: padding, match TF
    :return: output tensor
    """
    dtype = tf.float32

    input_size = int(input.get_shape()[3], )
    if w_std is None:
        w_std = 1.0 / np.sqrt(float(input_size * kernel_size * kernel_size))

    kernel = tf.get_variable('w_'+name,
                             [kernel_size, kernel_size, input_size, filters],
                             initializer=tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype),
                             dtype=dtype)
    output = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

    if addbias:
        biases = tf.get_variable('b_' + name, [filters], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(output, biases)
    return output


def linear_layer(input, output_size, name, w_mean=0.0, w_std=None):
    """
    Create variables and operator for a linear layer
    :param input: input tensor
    :param output_size: output size, number of hidden units
    :param name: variable name for linear weights and bias
    :param w_mean: mean of initializer for kernel weights
    :param w_std: standard deviation of initializer for kernel weights. Use 1/sqrt(input_param_count) if None.
    :return: output tensor
    """
    dtype = tf.float32

    if w_std is None:
        w_std = 1.0 / np.sqrt(float(np.prod(input.get_shape().as_list()[1])))

    w = tf.get_variable('w_' + name,
                        [input.get_shape()[1], output_size],
                        initializer=tf.truncated_normal_initializer(mean=w_mean, stddev=w_std, dtype=dtype),
                        dtype=dtype)

    b = tf.get_variable("b_" + name, [output_size], initializer=tf.constant_initializer(0.0))

    output = tf.matmul(input, w) + b

    return output


def conv_layers(input, conv_params, names, **kwargs):
    """ Build convolution layers from a list of descriptions.
        Each descriptor is a list: [kernel, hidden filters, activation]
    """
    output = input
    for layer_i in range(conv_params.shape[0]):
        kernelsize = int(conv_params[layer_i][0])
        hiddensize = int(conv_params[layer_i][1])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names+'_%d'%layer_i
        output = conv_layer(output, kernelsize, hiddensize, name, **kwargs)
        output = activation(output, conv_params[layer_i][2])
    return output


def fc_layers(input, conv_params, names, **kwargs):
    """ Build convolution layers from a list of descriptions.
        Each descriptor is a list: [size, _, activation]
    """
    output = input
    for layer_i in range(conv_params.shape[0]):
        size = int(conv_params[layer_i][0])
        if isinstance(names, list):
            name = names[layer_i]
        else:
            name = names+'_%d'%layer_i
        output = linear_layer(output, size, name, **kwargs)
        output = activation(output, conv_params[layer_i][-1])
    return output


def activation(tensor, activation_name):
    """
    Apply activation function to tensor
    :param tensor: input tensor
    :param activation_name: string that defines activation [lin, relu, tanh, sig]
    :return: output tensor
    """
    if activation_name in ['l', 'lin']:
        pass
    elif activation_name in ['r', 'relu']:
        tensor = tf.nn.relu(tensor)
    elif activation_name in ['t', 'tanh']:
        tensor = tf.nn.tanh(tensor)
    elif activation_name in ['s', 'sig']:
        tensor = tf.nn.sigmoid(tensor)
    elif activation_name in ['sm', 'smax']:
        tensor = tf.nn.softmax(tensor, dim=-1)
    else:
        raise NotImplementedError

    return tensor