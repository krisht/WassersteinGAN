import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist = input_data.read_data_sets('../data/', one_hot=True)


def def_weight(shape, name, coll_name):
    var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection(coll_name, var)
    return var


def def_bias(shape, name, coll_name):
    var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.constant_initializer(0.0))
    tf.add_to_collection(coll_name, var)
    return var


def get_loss(crit_loss, gen_loss):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(crit_loss, 'r--')
    plt.xlabel("Iterations")
    plt.ylabel("Critic Loss")
    plt.title("Iterations vs. Critic Loss")
    plt.subplot(212)
    plt.plot(gen_loss, 'g--')
    plt.xlabel("Iterations")
    plt.ylabel("Generator Loss")
    plt.title("Iterations vs. Generator Loss")
    plt.tight_layout()
    plt.show()


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
                 l_rate=5e-5,
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
        self.crit_real = self.critic(self.input, "real")
        self.crit_fake = self.critic(self.gen_sample, "fake")

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

            _, self.temp_gen_loss= self.sess.run([self.gen_optim, self.gen_loss],
                                                     feed_dict={self.output: random_noise(self.batch_size,
                                                                                          self.output_dims)})
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

        get_loss(self.crit_loss_arr, self.gen_loss_arr)

    def generator(self, z):
        G_W1 = def_weight([self.output_dims, 128], 'g_w1', 'gen_vars')
        G_b1 = def_bias([128], 'g_b1', 'gen_vars')

        G_W2 = def_weight([128, self.input_dims], 'g_w2', 'gen_vars')
        G_b2 = def_bias([self.input_dims], 'g_b2', 'gen_vars')

        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        g_prob = tf.nn.sigmoid(G_log_prob)
        return g_prob

    def critic(self, x, s):
        D_W1 = def_weight([self.input_dims, 128], 'd_w1'+s, 'crit_vars')
        D_b1 = def_bias([128], 'd_b1'+s, 'crit_vars')
        D_W2 = def_weight([128,1], 'd_w2'+s, 'crit_vars')
        D_b2 = def_bias([1], 'd_b2'+s, 'crit_vars')
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        out = tf.matmul(D_h1, D_W2) + D_b2
        return out


sess = tf.Session()
model = WGAN(sess=sess)
model.train()
