import numpy as np


def leaky_relu(x, slope=0.1, derivative=False):
    if derivative:
        v = np.ones(x.shape)
        v[x < 0.] = slope
        return v
    else:
        return np.maximum(slope*x, x)


def relu(x, derivative=False):
    if derivative:
        v = np.ones(x.shape)
        v[x<0.] = 0.
        return v
    else:
        return np.maximum(0, x)


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1. - sigmoid(x))
    else:
        return 1./(1. + np.exp(-x))