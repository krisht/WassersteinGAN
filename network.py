import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys


data_dir ='/data/images/'
chkpt_dir = '/chkpts/'
lrn_rate = 1e-4
bth_size = 64
dataset = 'celeba'

def prelu(x):



def gen(y_hat):
    y_hat = slim.fully_connected(yhat, 4*4*1024, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='gen_l1')
    y_hat = tf.reshape(y_hat, [bth_size, 4, 4, 1024]);

    # Conv Layer 1 for Generation

    y_hat = slim.convolution2d_transpose(y_hat, 512, 5, stride=2, normalizer=slim.batch_norm, activation_fn=tf.identity, scope='gen_l2')
    y_hat = tf.nn.relu(y_hat)


def cri(input_image, batch_size, reuse = False):


