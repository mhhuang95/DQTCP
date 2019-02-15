
import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
import sys
from env.sender import Sender
from Q_model import Q_network
from helpers.helpers import normalize, one_hot, softmax
from environment import Environment
import random


def create_env():
    uplink_trace = path.join(project_root.DIR, 'env', '12mbps.trace')
    downlink_trace = uplink_trace
    mahimahi_cmd = (
        'mm-delay 20 mm-link %s %s '
        '--downlink-queue=droptail --downlink-queue-args=packets=200' %
        (uplink_trace, downlink_trace))

    env = Environment(mahimahi_cmd)
    return env


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


class Learner(object):
    def __init__(self, env,restore_vars):

        self.batch_size = 32
        self.update_freq = 4
        self.y = 0.99
        self.tau = 0.001
        self.init = tf.global_variables_initializer()
        self.total_steps = 0
        self.num_episode = 10
        self.max_epLength = 10000
        self.myBuffer = experience_buffer()

        self.trainables = tf.trainable_variables()

        self.targetOps = self.updateTargetGraph(self.trainables, self.tau)

        self.rAll = 0
        self.jList = []
        self.rList = []


        self.aug_state_dim = env.state_dim + env.action_cnt
        self.action_cnt = env.action_cnt
        self.prev_action = env.action_cnt - 1




        with tf.variable_scope('global'):
            self.mainQN = Q_network(state_dim=self.aug_state_dim, action_cnt=self.action_cnt)
            self.targetQN = Q_network(state_dim=self.aug_state_dim, action_cnt=self.action_cnt)

        self.sess = tf.Session()

        saver = tf.train.Saver()
        saver.restore(self.sess, restore_vars)


        self.env = env

    def updateTargetGraph(self,tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        return op_holder

    def updateTarget(self,op_holder, sess):
        for op in op_holder:
            sess.run(op)


    def sample_action(self, state):
        if np.random.rand(1) < e:
            action = np.random.randint(0,self.env.action_cnt)
        else:
            norm_state = normalize(state)

            one_hot_action = one_hot(self.prev_action, self.action_cnt)
            aug_state = norm_state + one_hot_action

            # Get probability of each action from the local network.
            pi = self.mainQN
            feed_dict = {
                pi.state: [[aug_state]],
            }
            ops_to_run = [pi.action_probs]
            action_probs = self.sess.run(ops_to_run, feed_dict)

            # Choose an action to take
            action = np.argmax(action_probs[0][0])

        self.prev_action = action
        return action

    def update_model(self):


    def run(self):
        self.sess.run(self.init)

        for i in xrange(self.num_episode):
            episodeBuffer = experience_buffer()
            state = self.env.reset()
            done = False
            rall =0
            j = 0
            while j < self.max_epLength:
                j+=1
                action = self.sample_action(state)
                state_1, reward, done = self.env.step(action)
                self.total_steps += 1
                episodeBuffer.add(np.reshape(np.array([state,action,reward,state_1,done]),[1,5]))

                if self.total_steps % (self.update_freq) == 0:
                    trainBatch = self.myBuffer.sample(self.batch_size)
                    Q1 = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = self.sess.run(self.targetQN.Qout, feed_dict={self.targetQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[xrange(self.batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (self.y * doubleQ * end_multiplier)

                    _ = self.sess.run(self.mainQN.updateModel,
                                 feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), self.mainQN.targetQ: targetQ,
                                            self.mainQN.actions: trainBatch[:, 1]})

                    self.updateTarget(self.targetOps,self.sess)

                self.rAll += reward
                state = state_1

                if done == True:
                    break


            self.myBuffer.add(episodeBuffer.buffer)
            self.jList.append(j)
            self.rList.append(self.rAll)



    def cleanup(self):
        self.env.cleanup()











    def run(self):
        for episode_i in xrange(1, 3):
            sys.stderr.write('--- Episode %d\n' % episode_i)
            self.env.reset()

            # get an episode of experience
            self.env.rollout()

            # update model
            self.update_model()

    def update_model(self):
        sys.stderr.write('Updating model...\n')
