"""
Script to train QMDP-net and evaluate the learned policy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time
import numpy as np
import tensorflow as tf

import datafeed
from qmdpnet import QMDPNet, QMDPNetPolicy
from arguments import parse_args

try:
    import ipdb as pdb
except Exception:
    import pdb


def run_training(params):
    """
    Train qmdp-net.
    """
    # build dataflows
    datafile = os.path.join(params.path, "train/data.hdf5")
    train_feed = datafeed.Datafeed(params, filename=datafile, mode="train", max_env=params.training_envs)
    valid_feed = datafeed.Datafeed(params, filename=datafile, mode="valid", min_env=params.training_envs)

    # get cache for training data
    train_cache = train_feed.build_cache()

    df_train = train_feed.build_dataflow(params.batch_size, params.step_size, cache=train_cache)
    df_valid = valid_feed.build_dataflow(params.batch_size, params.step_size, cache=train_cache,
                                         restart_limit=10000)  # restart after full validation set

    df_train.reset_state()
    time.sleep(0.2)
    df_valid.reset_state()
    time.sleep(0.2)

    train_iterator = df_train.get_data()
    valid_iterator = df_valid.get_data()

    # built model into the default graph
    with tf.Graph().as_default():
        # build network for training
        network = QMDPNet(params, batch_size=params.batch_size, step_size=params.step_size)
        network.build_inference()  # build graph for inference including loss
        network.build_train(initial_lr=params.learning_rate)  # build training ops

        # build network for evaluation
        network_pred = QMDPNet(params, batch_size=1, step_size=1)
        network_pred.build_inference(reuse=True)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=100)  # if max_to_keep=0 will output useless log info

        # Get initialize Op
        init = tf.global_variables_initializer()

        # Create a TF session
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Run the Op to initialize variables
        sess.run(init)

        # load previously saved model
        if params.loadmodel:
            print ("Loading from "+params.loadmodel[0])
            loader = tf.train.Saver(var_list=tf.trainable_variables())
            loader.restore(sess, params.loadmodel[0])

        summary_writer = tf.summary.FileWriter(params.logpath, sess.graph)
        summary_writer.flush()

    epoch = -1
    best_epoch = 0
    no_improvement_epochs = 0
    patience = params.patience_first  # initial patience
    decay_step = 0
    valid_losses = []

    for epoch in range(params.epochs):
        training_loss = 0.0
        for step in range(train_feed.steps_in_epoch):
            data = train_iterator.next()
            feed_dict = {network.placeholders[i]: data[i] for i in range(len(network.placeholders))}

            _, loss, _ = sess.run([network.train_op, network.loss, network.update_belief_op],
                                  feed_dict=feed_dict)
            training_loss += loss

        # save belief and restore it after validation
        belief = sess.run([network.belief])[0]

        # accumulate loss over the enitre validation set
        valid_loss = 0.0
        for step in range(valid_feed.steps_in_epoch):  # params.validbatchsize
            data = valid_iterator.next()
            assert step > 0 or np.isclose(data[3], 1.0).all()
            feed_dict = {network.placeholders[i]: data[i] for i in range(len(network.placeholders))}
            loss, _ = sess.run([network.loss, network.update_belief_op], feed_dict=feed_dict)
            valid_loss += loss

        tf.assign(network.belief, belief)

        training_loss /= train_feed.steps_in_epoch
        valid_loss /= valid_feed.steps_in_epoch

        # print status
        lr = sess.run([network.learning_rate])[0]
        print('Epoch %d, lr=%f, training loss=%.3f, valid loss=%.3f' % (epoch, lr, training_loss, valid_loss))

        valid_losses.append(valid_loss)
        best_epoch = np.array(valid_losses).argmin()

        # save a checkpoint if needed
        if best_epoch == epoch or epoch == 0:
            best_model = saver.save(sess, os.path.join(params.logpath, 'model.chk'), global_step=epoch)
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # check for early stopping
        if no_improvement_epochs > patience:
            # finish training if learning rate decay steps reached
            if decay_step >= params.decaystep:
                break
            decay_step += 1
            no_improvement_epochs = 0

            # restore best model found so far
            saver.restore(sess, best_model)

            # decay learning rate
            sess.run(tf.assign(network.decay_step, decay_step))
            learning_rate = network.learning_rate.eval(session=sess)
            print("Decay step %d, lr = %f" % (decay_step, learning_rate))

            # use smaller patience for future iterations
            patience = params.patience_rest

    # Training done
    epoch += 1
    print("Training loop over after %d epochs" % epoch)

    # restore best model
    if best_epoch != epoch:
        print("Restoring %s from epoch %d" % (str(best_model), best_epoch))
        saver.restore(sess, best_model)

    # save best model
    checkpoint_file = os.path.join(params.logpath, 'final.chk')
    saver.save(sess, checkpoint_file)

    return checkpoint_file


def run_eval(params, modelfile):
    # built model into the default graph
    with tf.Graph().as_default():
        # build network for evaluation
        network = QMDPNet(params, batch_size=1, step_size=1)
        network.build_inference()

        # Create a saver for loading checkpoint
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        # Create a TF session
        os.environ["CUDA_VISIBLE_DEVICES"] = "" # use CPU
        sess = tf.Session(config=tf.ConfigProto())

        # load model from file
        saver.restore(sess, modelfile)

        # policy
        policy = QMDPNetPolicy(network, sess)

    # build dataflows
    eval_feed = datafeed.Datafeed(params, filename=os.path.join(params.path, "test/data.hdf5"), mode="eval")
    df = eval_feed.build_eval_dataflow(policy=policy, repeats=params.eval_repeats)
    df.reset_state()
    time.sleep(0.2)
    eval_iterator = df.get_data()

    print ("Evaluating %d samples, repeating simulation %d time(s)"%(params.eval_samples, params.eval_repeats))
    expert_results = []
    network_results = []
    for eval_i in range(params.eval_samples):
        res = eval_iterator.next()
        expert_results.append(res[:1]) # success, traj_len, collided, reward_sum
        network_results.append(res[1:])

    def print_results(results):
        results = np.concatenate(results, axis=0)
        succ = results[:,0]
        traj_len = results[succ > 0 ,1]
        collided = results[succ > 0, 2]
        reward = results[:, 3]
        print ("Success rate: %.3f  Trajectory length: %.1f  Collision rate: %.3f"%(
            np.mean(succ), np.mean(traj_len), np.mean(collided)))
    print ("Expert")
    print_results(expert_results)
    print ("QMDP-Net")
    print_results(network_results)


def main(arglist):
    params = parse_args(arglist)
    print(params)

    if params.epochs == 0:
        assert len(params.loadmodel) == 1
        modelfile = params.loadmodel[0]
    else:
        modelfile = run_training(params)

    run_eval(params, modelfile)


if __name__ == '__main__':
    main(sys.argv[1:])  # skip filename
