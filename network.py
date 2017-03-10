import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class WGAN:
    def __init__(self, input_size=10, clip_rate=[-0.01, 0.01], learning_rate=1e-4):
        self.sess = tf.Session()
        self.build_net(input_size=input_size, clip_rate=clip_rate, learning_rate=learning_rate)

    # Build the generator
    def generator(self, input_noise):
        with slim.arg_scope([slim.layers.fully_connected], biases_initializer=tf.constant_initializer(0),
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.layers.fully_connected(input_noise, 128, scope='gen_l1', activation_fn=tf.nn.relu)
            net = slim.layers.fully_connected(net, 784, scope='gen_l2', activation_fn=tf.nn.sigmoid)
            net = tf.reshape(net, [-1, 28, 28, 1])

            return net

    # Build critic
    def critic(self, input_image, reuse_scope=False):
        with slim.arg_scope([slim.layers.fully_connected], biases_initializer=tf.constant_initializer(0),
                            activation_fn=None, reuse=reuse_scope, weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.flatten(input_image)
            net = slim.layers.fully_connected(net, 128, scope='critic_l1', activation_fn=tf.nn.relu)
            net = slim.layers.fully_connected(net, 1, scope='critic_l2')
            return net

    # Sets up the training
    def build_net(self, input_size, clip_rate, learning_rate):
        # Input data
        self.input_size = input_size
        self.input_noise = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input_noise')
        self.real_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_images')

        # Generator output
        self.generated_MNIST = self.generator(self.input_noise)

        # Error values
        real_err = self.critic(self.real_images)
        fake_err = self.critic(self.generated_MNIST, reuse_scope=True)
        self.gen_err = tf.reduce_mean(fake_err)
        self.critic_err = tf.reduce_mean(real_err) - self.gen_err

        # For clipping
        variables = tf.trainable_variables()
        gen_var = [var for var in variables if 'gen' in var.name]
        critic_var = [var for var in variables if 'critic' in var.name]

        # Training operations
        self.clipper = [var.assign(tf.clip_by_value(var, clip_rate[0], clip_rate[1])) for var in critic_var]
        self.train_gen = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(-self.gen_err, var_list=gen_var)
        self.train_critic = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(-self.critic_err,
                                                                                            var_list=critic_var)

        self.sess.run(tf.global_variables_initializer())

    # Train a single iteration
    def train_iteration(self, critic_loops, images):
        # Train critic first
        critic_loss = 0
        batch_size = len(images)
        critic_loss = 0
        for critic_epoch in range(critic_loops):
            input_noise = np.random.uniform(0, 1.0, size=[batch_size, self.input_size])
            _, _, critic_loss = self.sess.run([self.train_critic, self.clipper, self.critic_err],
                                              feed_dict={self.input_noise: input_noise, self.real_images: images})
        # train generator
        input_noise = np.random.uniform(0, 1.0, size=[batch_size, self.input_size])
        _, gen_loss = self.sess.run([self.train_gen, self.gen_err], feed_dict={self.input_noise: input_noise})
        return float(critic_loss), float(gen_loss)

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
