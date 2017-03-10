import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist = input_data.read_data_sets('../data/', one_hot=True)

x,y = mnist.train.next_batch(1)

x = np.reshape(x, (1, 28, 28,1))

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# it's height, width in TF - not width, heig0ht

scale = 100
new_height = int(round(28* scale))
new_width = int(round(28 * scale))
# after TF version r0.11
resized = tf.image.resize_bicubic(X, [new_height, new_width])

print(resized)

tf.global_variables_initializer()

sess = tf.Session()

out = sess.run(resized, feed_dict={X: x})

print(out.shape)

plt.figure(1)

plt.imshow(out.reshape(new_height,new_width), cmap='Greys_r')


plt.figure(2)


plt.imshow(x.reshape(28, 28), cmap='Greys_r')

plt.show()