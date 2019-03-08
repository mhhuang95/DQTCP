import gym.spaces
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import random
import os


env = gym.make("Breakout-v0")
num_actions = 4
def test_env():
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        print(env.actions)
        observation, reward, done, info = env.step(action)

class Q_net():
    def __init__(self, h_size):
        self.scalarInput = tf.placeholder(shape=(None,100800),dtype=tf.float32)
        self.imagein = tf.reshape(self.scalarInput, shape=[-1, 210, 160, 3])
        self.conv1 = slim.conv2d(inputs=self.imagein, num_outputs=32, kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv1, num_outputs=h_size, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)

        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([380160, num_actions]))
        self.VW = tf.Variable(xavier_init([380160, 1]))
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

def processState(states):
    return np.reshape(states,[100800])


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
    batch_size = 32  # How many experiences to use for each training step.
    update_freq = 4  # How often to perform a training step.
    y = .99  # Discount factor on the target Q-values
    startE = 1  # Starting chance of random action
    endE = 0.1  # Final chance of random action
    annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
    num_episodes = 10000  # How many episodes of game environment to train network with.
    pre_train_steps = 10000  # How many steps of random actions before training begins.
    max_epLength = 50  # The max allowed length of our episode.
    load_model = False  # Whether to load a saved model.
    path = "./dqn"  # The path to save our model to.
    h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
    tau = 0.001  # Rate to update target network toward primary network

    tf.reset_default_graph()
    mainQN = Q_net(h_size)
    targetQN = Q_net(h_size)

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

        for i in range(num_episodes):
            episodeBuffer = experience_buffer()
            s = env.reset()
            s = processState(s)
            d = False
            rAll = 0
            j = 0

            while j < max_epLength:
                j += 1

                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = np.random.randint(0, 4)
                else:
                    a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                s1, r, d,_ = env.step(a)
                s1 = processState(s1)
                total_steps += 1
                episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size)
                        Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                        Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                        end_multiplier = -(trainBatch[:, 4] - 1)

                        doubleQ = Q2[range(batch_size), Q1]
                        targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                        # Update the network with our target values.
                        _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})

                        updateTarget(targetOps, sess)

                rAll += r
                s = s1

                if d == True:
                    break

            myBuffer.add(episodeBuffer.buffer)
            jList.append(j)
            rList.append(rAll)

            if i % 1000 == 0:
                saver.save(sess, path + '/model-' + str(i) + '.ckpt')
                print("Saved Model")
            if len(rList) % 10 == 0:
                print(total_steps, np.mean(rList[-10:]), e)
        saver.save(sess, path + '/model-' + str(i) + '.ckpt')
    print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")


if __name__ == "__main__":
    main()