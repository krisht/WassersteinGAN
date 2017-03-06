from network import WGAN
import numpy as np
import scipy.misc as science

test = WGAN()
test.train_iteration(1, 2, 100, [np.reshape(range(784), newshape=[28,28,1])])
images = test.generate_image(10)
for image, number in zip(images, range(len(images))):
    science.imsave("out" + str(number), image, 'JPEG')


