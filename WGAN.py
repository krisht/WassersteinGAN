import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import MLP
import bilinear

mnist = input_data.read_data_sets('../data/', one_hot=True)

crit_loss_arr = []
gen_loss_arr = []
pic_samples = 3


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
                 l_rate=5e-5,
                 n_iter=100000,
                 batch_size=16,
                 input_dims=784,
                 output_dims=10,
                 crit_train=5,
                 clip_val=0.01,
                 pic_samples=3,
                 generator=None,
                 critic=None):
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

        self.gen_sample = generator(self.output, output_dims=output_dims, input_dims=input_dims)
        self.crit_real = critic(self.input, input_dims=input_dims)
        self.crit_fake = critic(self.gen_sample, input_dims=input_dims, reuse_scope=True)

        self.crit_loss = tf.reduce_mean(self.crit_real) - tf.reduce_mean(self.crit_fake)
        self.gen_loss = -tf.reduce_mean(self.crit_fake)
        train_vars = tf.trainable_variables()
        gen_var = [x for x in train_vars if 'gen' in x.name]
        crit_var = [x for x in train_vars if 'crit' in x.name]
        self.crit_optim = (tf.train.RMSPropOptimizer(learning_rate=self.l_rate)) \
            .minimize(-self.crit_loss, var_list=crit_var)
        self.gen_optim = (tf.train.RMSPropOptimizer(learning_rate=self.l_rate)) \
            .minimize(self.gen_loss, var_list=gen_var)
        self.crit_clipper = [p.assign(tf.clip_by_value(p, -clip_val, clip_val)) for p in crit_var]

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


try:
    sess = tf.Session()
    model = WGAN(sess=sess, pic_samples=pic_samples, generator=bilinear.generator, critic=bilinear.critic, n_iter=50000)
    model.train()
    get_loss(crit_loss_arr, gen_loss_arr)
except (KeyboardInterrupt, SystemExit, SystemError):
    get_loss(crit_loss_arr, gen_loss_arr)
get_loss(crit_loss_arr, gen_loss_arr)