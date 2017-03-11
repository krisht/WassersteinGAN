import tensorflow as tf
import tensorflow.contrib.slim as slim


def generator(input, output_dims, input_dims):
    with slim.arg_scope([slim.layers.fully_connected, slim.layers.convolution], outputs_collections='gen_vars'):
        net = slim.layers.fully_connected(input, num_outputs=20, scope='fc1')
        net = slim.layers.fully_connected(net, num_outputs=25, scope='fc2')
        net = tf.reshape(net, shape=[-1, 5, 5, 1])
        net = tf.image.resize_bilinear(net, size=[15, 15])
        net = slim.layers.convolution(net, num_outputs=10, kernel_size=3, stride=1, scope='conv1')
        net = tf.image.resize_bilinear(net, size=[28, 28])
        net = slim.layers.convolution(net, num_outputs=1, kernel_size=3, stride=1, scope='conv2')
        net = tf.nn.sigmoid(net)
        return net


def critic(input, input_dims, reuse_scope=False):
    with slim.arg_scope([slim.layers.fully_connected, slim.layers.convolution], outputs_collections='crit_vars',
                        reuse=reuse_scope):
        net = tf.reshape(input, shape=[-1, 28, 28, 1])
        net = slim.layers.convolution(net, num_outputs=20, kernel_size=3, stride=1, scope='conv1')
        net = slim.layers.convolution(net, num_outputs=40, kernel_size=3, stride=1, scope='conv2')
        net = slim.flatten(net)
        net = slim.layers.fully_connected(net, num_outputs=150, scope='fc1')
        net = slim.layers.fully_connected(net, num_outputs=50, scope='fc2')
        net = slim.layers.fully_connected(net, num_outputs=1, activation_fn=None, scope='fc3')
        return net
