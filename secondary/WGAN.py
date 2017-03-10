import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist = input_data.read_data_sets('../data/', one_hot=True)

crit_loss_arr = []
gen_loss_arr = []
pic_samples = 4


def def_weight(shape, name, coll_name, reuse_scope=True):
    with tf.variable_scope('weights', reuse=reuse_scope):
        var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection(coll_name, var)
        return var


def def_bias(shape, name, coll_name, reuse_scope=True):
    with tf.variable_scope('biases', reuse=reuse_scope):
        var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.constant_initializer(0.0))
        tf.add_to_collection(coll_name, var)
        return var

def prelu(_x, name, reuse_scope=False):
    with tf.variable_scope('prelu', reuse=reuse_scope):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        tf.add_to_collection('model_vars', alphas)

        return pos + neg


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
    fig = plt.figure(figsize=(pic_samples, pic_samples))
    gs = gridspec.GridSpec(pic_samples, pic_samples)
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
                 n_iter=100000,
                 batch_size=16,
                 input_dims=784,
                 output_dims=10,
                 crit_train=5,
                 clip_val = 0.01,
                 pic_samples=3):
        self.n_iter = n_iter
        self.crit_train = crit_train
        self.sess = sess
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.pic_samples = pic_samples
        self.clip_val = clip_val
        self.input = tf.placeholder(tf.float32, shape=[None, self.input_dims])
        self.output = tf.placeholder(tf.float32, shape=[None, self.output_dims])
        self.l_rate = l_rate

        # Build Model

        self.gen_sample = self.generator(self.output)
        self.crit_real = self.critic(self.input)
        self.crit_fake = self.critic(self.gen_sample, reuse_scope=True)

        self.crit_loss = tf.reduce_mean(self.crit_real) - tf.reduce_mean(self.crit_fake)
        self.gen_loss = -tf.reduce_mean(self.crit_fake)

        self.crit_optim = (tf.train.RMSPropOptimizer(learning_rate=self.l_rate)) \
            .minimize(-self.crit_loss, var_list=tf.get_collection('crit_vars'))
        self.gen_optim = (tf.train.RMSPropOptimizer(learning_rate=self.l_rate)) \
            .minimize(self.gen_loss, var_list=tf.get_collection('gen_vars'))
        self.crit_clipper = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in tf.get_collection('crit_vars')]

        self.temp_crit_loss = 0
        self.temp_gen_loss = 0

    def wasserstein_algorithm(self):
        kk = 0
        for iterr in range(self.n_iter):
            for _ in range(self.crit_train):
                samples, _ = mnist.train.next_batch(self.batch_size)
                _, self.temp_crit_loss, _ = self.sess.run([self.crit_optim, self.crit_loss, self.crit_clipper],
                                                          feed_dict={self.input: samples,
                                                                     self.output: random_noise(self.batch_size,
                                                                                               self.output_dims)})
            _, self.temp_gen_loss = self.sess.run([self.gen_optim, self.gen_loss],
                                                  feed_dict={self.output: random_noise(self.batch_size,
                                                                                       self.output_dims)})
            crit_loss_arr.append(self.temp_crit_loss)
            gen_loss_arr.append(self.temp_gen_loss)

            if iterr % 100 == 0:
                print("Iteration: {:5}, Critic Loss: {:5.5}, Gen Loss: {:5.5}".format(iterr, self.temp_crit_loss,
                                                                                      self.temp_gen_loss))
                if iterr % 1000 == 0:
                    samples = self.sess.run(self.gen_sample,
                                            feed_dict={self.output: random_noise(self.pic_samples**2,
                                                                                 self.output_dims)})
                    fig = get_samples(samples)
                    plt.savefig('outputs/{}.png'.format(str(kk).zfill(3)), bbox_inches='tight')
                    kk += 1
                    plt.close(fig)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        if not os.path.exists('outputs/'):
            os.makedirs('outputs/')
        self.wasserstein_algorithm()

    def generator(self, z):
        G_W1 = def_weight([self.output_dims, 128], 'g_w1', 'gen_vars', reuse_scope=False)
        G_b1 = def_bias([128], 'g_b1', 'gen_vars', reuse_scope=False)

        G_l1 = prelu(tf.matmul(z, G_W1) + G_b1, 'prelu1')

        G_W2 = def_weight([128, 256], 'g_w2', 'gen_vars', reuse_scope=False)
        G_b2 = def_bias([256], 'g_b2', 'gen_vars', reuse_scope=False)

        G_l2 = prelu(tf.matmul(G_l1, G_W2) + G_b2, 'prelu2')

        G_W3 = def_weight([256, self.input_dims], 'g_w3', 'gen_vars', reuse_scope=False)
        G_b3 = def_bias([self.input_dims], 'g_b3', 'gen_vars', reuse_scope=False)

        G_log_prob = tf.matmul(G_l2, G_W3) + G_b3
        g_prob = tf.nn.sigmoid(G_log_prob)
        return g_prob

    def critic(self, x, reuse_scope=False):
        D_W1 = def_weight([self.input_dims, 128], 'd_w1', 'crit_vars', reuse_scope=reuse_scope)
        D_b1 = def_bias([128], 'd_b1', 'crit_vars', reuse_scope=reuse_scope)

        D_l1 = prelu(tf.matmul(x, D_W1) + D_b1, 'prelu3', reuse_scope=reuse_scope)

        D_W2 = def_weight([128, 256], 'd_w2', 'crit_vars', reuse_scope=reuse_scope)
        D_b2 = def_weight([256], 'd_b2', 'crit_vars', reuse_scope=reuse_scope)

        D_l2 = prelu(tf.matmul(D_l1, D_W2) + D_b2, 'prelu4', reuse_scope=reuse_scope)

        D_W3 = def_weight([256, 1], 'd_w3', 'crit_vars', reuse_scope=reuse_scope)
        D_b3 = def_weight([1], 'd_b3', 'crit_vars', reuse_scope=reuse_scope)

        out = tf.matmul(D_l2, D_W3) + D_b3
        return out


try:
    sess = tf.Session()
    model = WGAN(sess=sess, pic_samples=pic_samples)
    model.train()
except (KeyboardInterrupt, SystemExit, SystemError):
    get_loss(crit_loss_arr, gen_loss_arr)