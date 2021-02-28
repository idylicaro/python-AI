from random import *
import numpy as np
import matplotlib.pyplot as plt

LEARN_RATING = 0.1

weight1 = random()
weight2 = random()
weight_bias = random()


# input1 = int(input('Write the input (1)'))
# input2 = int(input('Write the input (2)'))
# bias = 1
#
# error = 1

def hypothesis(_x, w0, w1):
    return w0 + w1 * _x


def sigmoid(_x):
    return 1 / (1 + np.exp(-_x))


def delta_output(_error, dev):
    return _error * dev


def partial_derivative(sig):
    return sig * (1 - sig)


def Thiperbolica(_x):
    return (np.exp(_x) - np.exp(-_x)) / (np.exp(_x) + np.exp(-_x))


x = np.arange(-10, 10, step=1)

h = hypothesis(x, weight1, weight2)

total_h = np.arange(0, 20)
points_Y = [4, 5, 5]
points_X = [5, 3, 4]

plt.scatter(points_X, points_Y)
plt.plot(total_h, sigmoid(h))
plt.show()
