import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import random
import os

num_actions = 5

class Q_network(object):
    def __init__(self,):
        self.state = tf.placeholder(shape=[None, 5], dtype=tf.int32)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.fc1w = tf.Variable(xavier_init([5, 64]))
        self.fc1b = tf.Variable(xavier_init([None,64]))
        self.fc1 = tf.matmul(self.state, self.fc1w) + self.fc1b

        self.fc2w = tf.Variable(xavier_init([64, 64]))
        self.fc2b = tf.Variable(xavier_init([None, 64]))
        self.fc2 = tf.matmul(self.fc1, self.fc2w) + self.fc2b

        self.fc3w = tf.Variable(xavier_init([64,128]))
        self.fc3b = tf.Variable(xavier_init([None, 128]))
        self.fc3 = tf.matmul(self.fc2, self.fc3w) + self.fc3b

        self.streamAC, self.streamVC = tf.split(self.fc3, 2, 0)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        self.AW = tf.Variable(xavier_init([64, num_actions]))
        self.VW = tf.Variable(xavier_init([64, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)



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


def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def main():
    batch_size = 32
    update_freq = 4  # How often to perform a training step.
    y = .99  # Discount factor on the target Q-values
    startE = 1  # Starting chance of random action
    endE = 0.1  # Final chance of random action
    annealing_steps = 10000.  # How many steps of training to reduce startE to endE.

    pre_train_steps = 10000  # How many steps of random actions before training begins.
    max_epLength = 50  # The max allowed length of our episode.
    load_model = False  # Whether to load a saved model.
    path = "./dqn"  # The path to save our model to.
    tau = 0.001  # Rate to update target network toward primary network

    tf.reset_default_graph()
    mainQN = Q_network()
    targetQN = Q_network()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    myBuffer = experience_buffer()

    e = startE
    stepDrop = (startE - endE) / annealing_steps

    jList = []
    rList = []
    total_steps = 0

    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        sess.run(init)
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            s = env.reset()
            rAll = 0

            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.state: [s]})[0]
            s1,r = env.step(a)
            total_steps += 1
            myBuffer.add(np.array([s,a,r,s1]))

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.state: np.vstack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, feed_dict={mainQN.state: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,mainQN.actions: trainBatch[:, 1]})

                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s = s1
