from network import WGAN
import numpy as np
from scipy.misc import imsave, imshow
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

mnist = input_data.read_data_sets("data/", one_hot=True)

test = WGAN()

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def train(nEpoch, batch_size):
    for kk in range(nEpoch):
        print("Epoch:", kk)
        for samp in range(int(500/batch_size)):
            images, _  = mnist.train.next_batch(batch_size)
            images = images.reshape((batch_size, 28, 28, 1))
            test.train_iteration(10, images)
        samples = np.array(test.generate_image(16))
        fig = plot(samples)
        plt.savefig("{}.png".format(str(kk).zfill(3)), bbox_inches='tight')
        plt.close(fig)


train(50, 10)


