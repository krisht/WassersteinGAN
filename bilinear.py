import tensorflow as tf
import tensorflow.contrib.slim as slim


def generator(input, output_dims, input_dims):
    with slim.arg_scope([slim.layers.fully_connected, slim.layers.convolution], outputs_collections='gen_vars'):
        net = slim.layers.fully_connected(input, num_outputs=512, scope='gen_fc1', activation_fn=None)
        net = slim.layers.batch_norm(net)
        net = tf.reshape(net, shape=[-1, 2, 2, 1])
        net = tf.nn.relu(net)
        net = tf.image.resize_bilinear(net, size=[4, 4])
        net = slim.layers.convolution(net, num_outputs=20, kernel_size=3, stride=1, scope='gen_conv1', activation_fn=None)
        net = slim.layers.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.layers.convolution(net, num_outputs=20, kernel_size=3, stride=1, scope='gen_conv2')
        net = tf.image.resize_bilinear(net, size=[8,8])
        net = slim.layers.convolution(net, num_outputs=20, kernel_size=3, stride=1, scope='gen_conv3', activation_fn=None)
        net = slim.layers.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.layers.convolution(net, num_outputs=20, kernel_size=3, stride=1, scope='gen_conv4')
        net = tf.image.resize_bilinear(net, size=[16,16])
        net = slim.layers.convolution(net, num_outputs=20, kernel_size=3, stride=1, scope='gen_conv5', activation_fn=None)
        net = slim.layers.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.layers.convolution(net, num_outputs=20, kernel_size=3, stride=1, scope='gen_conv6')
        net = tf.image.resize_bilinear(net, size=[28,28])
        net = slim.layers.convolution(net, num_outputs=1, kernel_size=3, stride=1, scope='gen_conv7')
        net = tf.nn.sigmoid(net)
        return net


def critic(input, input_dims, reuse_scope=False):
    with slim.arg_scope([slim.layers.fully_connected, slim.layers.convolution], outputs_collections='crit_vars',
                        reuse=reuse_scope):
        net = tf.reshape(input, shape=[-1, 28, 28, 1])
        net = slim.layers.convolution(net, num_outputs=64, kernel_size=2, stride=1, scope='crit_conv1')
        net = slim.layers.batch_norm(net)
        net = slim.layers.convolution(net, num_outputs=128, kernel_size=3, stride=1, scope='crit_conv2')
        net = slim.layers.max_pool2d(net, kernel_size=2)
        net = slim.flatten(net)
        net = slim.layers.fully_connected(net, num_outputs=150, scope='crit_fc1')
        net = slim.layers.fully_connected(net, num_outputs=50, scope='crit_fc2')
        net = slim.layers.fully_connected(net, num_outputs=1, activation_fn=None, scope='crit_fc3')
        return net
