

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from Q_model import Q_network
from helpers.helpers import normalize, one_hot, softmax

class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        self.aug_state_dim = state_dim + action_cnt
        self.action_cnt = action_cnt
        self.prev_action = action_cnt - 1

        with tf.variable_scope('global'):
            self.model = Q_network(state_dim = self.aug_state_dim, action_cnt = self.action_cnt)

        self.mainQN = Q_network()


        self.sess = tf.Session()

        saver = tf.train.Saver(self.model.trainable_vars)
        saver.restore(self.sess, restore_vars)


    def sample_action(self, state):
        norm_state = normalize(state)

        one_hot_action = one_hot(self.prev_action, self.action_cnt)
        aug_state = norm_state + one_hot_action

        # Get probability of each action from the local network.
        pi = self.mainQN
        feed_dict = {
            pi.input: [[aug_state]]
        }
        ops_to_run = pi.action_probs
        action_probs = self.sess.run(ops_to_run, feed_dict)

        # Choose an action to take
        action = np.argmax(action_probs[0][0])
        self.prev_action = action

        return action

    def Q_policy(self, state):
        norm_state = normalize(state)

        one_hot_action = self.prev_action, self.action_cnt

