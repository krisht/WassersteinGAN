from network import WGAN
import numpy as np
from scipy.misc import imsave, imshow
from tensorflow.examples.tutorials.mnist import input_data
import warnings

warnings.filterwarnings('ignore')

mnist = input_data.read_data_sets("data/", one_hot=True)

test = WGAN()

def genPics(nPics):
    images = test.generate_image(10)
    for image, number in zip(images, range(len(images))):
        imsave("out" + str(number),  image, 'JPEG')


def train(nEpoch, batch_size):
    for kk in range(nEpoch):
        for samp in range(int(500/batch_size)):
            print("Epoch:", kk, "Sample:", samp)
            images, _  = mnist.train.next_batch(batch_size)
            images = images.reshape((batch_size, 28, 28, 1))
            test.train_iteration(10, images)
        genPics(10)


train(5, 10)


