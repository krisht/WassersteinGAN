import tensorflow as tf
import tensorflow.contrib.slim as slim

def generator(input, output_dims, input_dims):
    net = slim.layers.fully_connected(input, num_outputs=20)
    net = slim.layers.fully_connected(net, num_outputs=25)
    net = tf.reshape(net, shape=[None, 5, 5, 1])
    net = tf.image.resize_bilinear(net, size=[15,15])
    net = slim.layers.convolution(net, num_outputs=10)
    net = tf.image.resize_bilinear(net, size=[28,28])
    return net


def critic(input, input_dims, reuse_scope=False):
    net = tf.reshape(input, shape=[None, 28, 28, 1])


