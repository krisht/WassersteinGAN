from network import WGAN
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import random
from scipy.misc import imsave

mnist = read_data_sets("mnist_data/", one_hot=True)
train = mnist[0]
validate = mnist[1]
test = mnist[2]

poopoo_images = np.concatenate((train.images, validate.images, test.images))

epochs = 6000
batch_size = 256
test = WGAN(clip_rate=[-0.5, 0.5])
images = [np.reshape(x, newshape=[28,28,1]) for x in poopoo_images]

for epoch in range(1, epochs+1):
    print("Epoch " + str(epoch))
    sample = random.sample(images, batch_size)
    test.train_iteration(5, sample)

result = test.generate_image(10)
for image, number in zip(result, range(10)):
    imsave("out" + str(number), image, 'JPEG')
