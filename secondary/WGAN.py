import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist = input_data.read_data_sets('../data/', one_hot=True)


def def_weight(shape, name, coll_name):
    var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer)
    tf.add_to_collection(coll_name, var)
    return var


def def_bias(shape, name, coll_name):
    var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.constant_initializer(0.0))
    tf.add_to_collection(coll_name, var)
    return var


def get_samples(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def random_noise(m, n):
    return np.random.uniform(-1, 1, size=[m, n])


class WGAN(object):
    def __init__(self, sess,
                 l_rate=1e-4,
                 n_iter=1000000,
                 batch_size=100,
                 input_dims=784,
                 output_dims=10,
                 crit_train=5):
        self.n_iter = n_iter
        self.crit_train = crit_train
        self.sess = sess
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input = tf.placeholder(tf.float32, shape=[None, self.input_dims])
        self.output = tf.placeholder(tf.float32, shape=[None, self.output_dims])
        self.l_rate = l_rate

        self.crit_loss_arr = []
        self.gen_loss_arr = []

        # Build Model

        self.gen_sample = self.generator(self.output)
        self.crit_real = self.critic(self.input)
        self.crit_fake = self.critic(self.gen_sample)

        self.crit_loss = tf.reduce_mean(self.crit_real) - tf.reduce_mean(self.crit_fake)
        self.gen_loss = -tf.reduce_mean(self.crit_fake)

        self.crit_optim = (tf.train.RMSPropOptimizer(learning_rate=self.l_rate)) \
            .minimize(-self.crit_loss, var_list=tf.get_collection('crit_vars'))
        self.gen_optim = (tf.train.RMSPropOptimizer(learning_rate=self.l_rate)) \
            .minimize(self.gen_loss, var_list=tf.get_collection('gen_vars'))

        self.crit_clipper = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in tf.get_collection('crit_vars')]

        self.temp_crit_loss = 0
        self.temp_gen_loss = 0

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        if not os.path.exists('out/'):
            os.makedirs('out/')
        kk = 0
        for epoch in range(self.n_iter):
            for _ in range(self.crit_train):
                x, _ = mnist.train.next_batch(self.batch_size)
                _, self.temp_crit_loss, _ = self.sess.run([self.crit_optim, self.crit_loss, self.crit_clipper],
                                                          feed_dict={self.input: x,
                                                                     self.output: random_noise(self.batch_size,
                                                                                                    self.output_dims)})

            _, self.temp_gen_loss, _ = self.sess.run([self.gen_optim, self.gen_loss],
                                                     feed_dict={self.output: random_noise(self.batch_size, self.output_dims)})
            self.crit_loss_arr.append(self.temp_crit_loss)
            self.gen_loss_arr.append(self.temp_gen_loss)

            if epoch % 100 == 0:
                print("Epoch: {}, Critic Loss: {:0.5}, Gen Loss: {:0.5}".format(epoch, self.temp_crit_loss,
                                                                                self.temp_gen_loss))

                if epoch % 1000 == 0:
                    samples = self.sess.run(self.gen_sample,
                                            feed_dict={self.output: random_noise(16, self.output_dims)})
                    fig = get_samples(samples)
                    plt.savefig('out/{}.png'.format(str(kk).zfill(3)), bbox_inches='tight')
                    kk += 1
                    plt.close(fig)

        print(self.crit_loss_arr)
        print(self.gen_loss_arr)

    def generator(self, z):
        # Generator Model
        g_prob = 1
        return g_prob

    def critic(self, x):
        # Critic model
        out = 1
        return out


if __name__ == "__main__":
    sess = tf.Session()
    model = WGAN(sess=sess)
    model.train()
