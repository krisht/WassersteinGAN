import tensorflow as tf
import tensorflow.contrib.slim as slim

data_dir = '/data/images/'
chkpt_dir = '/chkpts/'
lrn_rate = 1e-4
bth_size = 64
dataset = 'celeba'


def gen(gen_img):
    gen_img = slim.fully_connected(gen_img, 4 * 4 * 1024, normalizer_fn=slim.batch_norm, activation_fn=tf.identity,
                                   scope='gen_l1')
    gen_img = tf.reshape(gen_img, [bth_size, 4, 4, 1024])

    # Deconv Layer 1 for Generation

    gen_img = slim.convolution2d_transpose(gen_img, 512, 5, stride=2, normalizer=slim.batch_norm,
                                           activation_fn=tf.identity, scope='gen_l2')
    gen_img = tf.nn.relu(gen_img)

    # Deconv Layer 2

    gen_img = slim.convolution2d_transpose(gen_img, 256, 5, stride=2, normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.identity, scope='gen_l3')
    gen_img = tf.nn.relu(gen_img)

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
        y_hat = tf.nn.relu(y_hat)

        # Conv Layer 2

        y_hat = slim.convolution(y_hat, 128, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity,
                                 scope='crit_l2')
        y_hat = tf.nn.relu(y_hat)

        # Conv Layer 3

        y_hat = slim.convolution(y_hat, 256, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity,
                                 scope='crit_l3')
        y_hat = slim.relu(y_hat)

        # Conv Layer 4

        y_hat = slim.convolution(y_hat, 256, 5, stride=2, normalizer=slim.batch_norm, activation_fn=tf.identity,
                                 scope='crit_l4')
        y_hat = slim.convolution(y_hat, 1, 4, stride=2, activation_fn=tf.identity, scope='crit_l5')

        tf.add_to_collection('model_vars', y_hat)

        return y_hat
