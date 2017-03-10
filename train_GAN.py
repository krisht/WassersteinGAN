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

epochs = 20000
batch_size = 16
test = WGAN(clip_rate=[-0.01, 0.01])
images = [np.reshape(x, newshape=[28,28,1]) for x in poopoo_images]

for epoch in range(epochs):
    sample = random.sample(images, batch_size)
    critic_loss, gen_loss = test.train_iteration(5, sample)
    if epoch%100 == 0:
        print("Epoch: {:5} Generator: {:5.5} Critic: {:5.5}".format(epoch, gen_loss, critic_loss))
    if epoch%1000 == 0:
        validate = test.generate_image(10)
        for image, number in zip(validate, range(10)):
            imsave(str(epoch)+"out" + str(number), image, 'JPEG')

result = test.generate_image(10)
for image, number in zip(result, range(10)):
    imsave("out" + str(number), image, 'JPEG')
