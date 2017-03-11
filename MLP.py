import tensorflow as tf
import tensorflow.contrib.slim as slim


def generator(z, output_dims, input_dims):
    net = slim.layers.fully_connected(z, num_outputs=128, scope='gen_fc1')
    net = slim.layers.fully_connected(net, num_outputs=256, scope='gen_fc2')
    net = slim.layers.fully_connected(net, num_outputs=input_dims, scope='gen_fc3', activation_fn=None)
    net = tf.nn.sigmoid(net)
    return net

def critic(x, input_dims, reuse_scope=False):
    with slim.arg_scope([slim.layers.fully_connected], reuse=reuse_scope):
        net = slim.layers.fully_connected(x, num_outputs=128, scope='crit_fc1')
        net = slim.layers.fully_connected(net, num_outputs=256, scope='crit_fc2')
        net = slim.layers.fully_connected(net, num_outputs=1, scope='crit_fc3', activation_fn=None)
        return net
