import tensorflow as tf
import tensorflow.contrib.slim as slim

data_dir = '/data/images/'
chkpt_dir = '/chkpts/'
lrn_rate = 1e-4
bth_size = 64
dataset = 'celeba'

def prelu(_x, name):
    alphas = tf.get_variable(name,  _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas*(_x - abs(_x)) * 0.5

    tf.add_to_collection('model_vars', alphas)

    return pos + neg


def gen(gen_img):
    gen_img = slim.fully_connected(gen_img, 4 * 4 * 1024, normalizer_fn=slim.batch_norm, activation_fn=tf.identity,
                                   scope='gen_l1')
    gen_img = tf.reshape(gen_img, [bth_size, 4, 4, 1024])

    # Deconv Layer 1 for Generation

    gen_img = slim.convolution2d_transpose(gen_img, 512, 5, stride=2, normalizer=slim.batch_norm,
                                           activation_fn=tf.identity, scope='gen_l2')
    gen_img = prelu(gen_img, 'gen_l2_prelu')

    # Deconv Layer 2

    gen_img = slim.convolution2d_transpose(gen_img, 256, 5, stride=2, normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.identity, scope='gen_l3')
    gen_img = prelu(gen_img, 'gen_l3_prelu')

    # Deconv Layer 3

    gen_img = slim.convolution2d_transpose(gen_img, 128, 5, stride=2, normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.identity, scope='gen_l4')
    gen_img = tf.nn.tanh(gen_img)

    tf.add_to_collection('model_vars', gen_img)

    return gen_img


def cri(input_image, reuse_scope=False):
    sc = tf.get_variable_scope()
    with tf.variable_scope(sc, reuse=reuse_scope):
        # Conv Layer 1

        y_hat = slim.convolution(input_image, 64, 5, stride=2, activation_fn=tf.identity, scope='crit_l1')
        y_hat = prelu(y_hat, 'crit_l1_prelu')

        # Conv Layer 2

        y_hat = slim.convolution(y_hat, 128, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity,
                                 scope='crit_l2')
        y_hat = prelu(y_hat, 'crit_l2_prelu')

        # Conv Layer 3

        y_hat = slim.convolution(y_hat, 256, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity,
                                 scope='crit_l3')
        y_hat = prelu(y_hat, 'crit_l3_prelu')

        # Conv Layer 4

        y_hat = slim.convolution(y_hat, 256, 5, stride=2, normalizer=slim.batch_norm, activation_fn=tf.identity,
                                 scope='crit_l4')
        y_hat = slim.convolution(y_hat, 1, 4, stride=2, activation_fn=tf.identity, scope='crit_l5')

        tf.add_to_collection('model_vars', y_hat)

        return y_hat
