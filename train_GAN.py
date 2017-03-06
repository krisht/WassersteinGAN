from network import WGAN
import numpy as np

test = WGAN()
test.train_iteration(1, 2, 100, [np.reshape(range(784), newshape=[28,28,1])])



