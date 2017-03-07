import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def prelu(_x, name, reuse_scope=False):
    with tf.variable_scope('prelu', reuse=reuse_scope):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        tf.add_to_collection('model_vars', alphas)

        return pos + neg


class WGAN:
    def __init__(self, input_size=100, clip_rate=[-1.0, 1.0], learning_rate=5e-5):
        self.sess = tf.Session()
        self.build_net(input_size=input_size, clip_rate=clip_rate, learning_rate=learning_rate)

    # Build the generator
    def generator(self, input_noise):
        # The output of the transpose convolution is input * stride here
        with slim.arg_scope([slim.layers.fully_connected, slim.layers.convolution2d_transpose],
                            normalizer_fn=slim.batch_norm, activation_fn=tf.identity):
            bs = tf.shape(input_noise)[0]

            net = slim.layers.fully_connected(input_noise, 1024, weights_regularizer=slim.l2_regularizer(2, 5e-5), activation_fn=tf.identity)



            net = slim.layers.fully_connected(input_noise, 7 * 7 * 24,
                                              scope='gen_l1')

            net = tf.reshape(net, [-1, 7, 7, 24])

            # Deconv Layer 1 for Generation

            net = slim.layers.convolution2d_transpose(net, 40, 5, stride=2, scope='gen_l2')
            net = prelu(net, 'gen_l2')

            # Deconv Layer 2

            net = slim.layers.convolution2d_transpose(net, 1, 5, stride=2, scope='gen_l3')

            net = 255*tf.nn.sigmoid(net)

            tf.add_to_collection('model_vars', net)

            return net

    # Build critic
    def critic(self, input_image, reuse_scope=False):
        with slim.arg_scope([slim.layers.convolution], activation_fn=tf.identity, reuse=reuse_scope):

            bs = tf.shape(input_image)[0]
            x = tf.reshape(x, [bs, 28, 28, 1])

            yhat = slim.layers.convolution2d(x, 64, [4,4], [2,2], activation_fn=tf.identity)

            yhat = prelu(yhat, "crit_prelu_1")

            yhat = slim.layers.convolution2d(yhat, 128, [4,4], [2,2], activation_fn=tf.identity)

            yhat = prelu(yhat, "crit_prelu_2")

            yhat = slim.layers.flatten(yhat)

            yhat = slim.layers.fully_connected(yhat, 1024, activation_fn=tf.identity)

            yhat = prelu(yhat, "crit_prelu_3")

            yhat = slim.layers.fully_connected(yhat, 1, activation_fn=tf.identity)

            tf.add_to_collection('model_vars', yhat)

            return yhat

    # Sets up the training
    def build_net(self, input_size, clip_rate, learning_rate):
        # Input data
        self.input_size = input_size
        self.input_noise = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input_noise')
        self.real_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_images')

        # Generator output
        self.generated_MNIST = self.generator(self.input_noise)

        # Error values
        fake_err = self.critic(self.generated_MNIST)
        real_err = self.critic(self.real_images, reuse_scope=True)
        critic_err = tf.reduce_mean(real_err - fake_err)
        gen_err = tf.reduce_mean(fake_err)

        # For clipping
        variables = tf.trainable_variables()
        gen_var = [var for var in variables if 'gen' in var.name]
        critic_var = [var for var in variables if 'critic' in var.name]

        # Training operations
        self.clipper = [var.assign(tf.clip_by_value(var, clip_rate[0], clip_rate[1])) for var in critic_var]
        self.train_gen = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(gen_err, var_list=gen_var)
        self.train_critic = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(critic_err,
                                                                                            var_list=critic_var)

        self.sess.run(tf.global_variables_initializer())

    # Train a single iteration
    def train_iteration(self, critic_loops, images):
        # Train critic first
        batch_size = len(images)
        for critic_epoch in range(critic_loops):
            input_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.input_size])
            self.sess.run(self.train_critic, feed_dict={self.input_noise: input_noise, self.real_images: images})
            self.sess.run(self.clipper)

        # train generator
        input_noise = np.random.normal(-1.0, 1.0, size=[batch_size, self.input_size])
        self.sess.run(self.train_gen, feed_dict={self.input_noise: input_noise})

    # Generates an image
    def generate_image(self, num_images):
        input_noise = np.random.normal(-1.0, 1.0, size=[num_images, self.input_size])
        output = self.sess.run(self.generated_MNIST, feed_dict={self.input_noise: input_noise})
        images = []
        for poopoo in output:
            image = []
            for row in poopoo:
                row_corrected = [col[0] for col in row]
                image.append(row_corrected)
            images.append(image)
        return images
