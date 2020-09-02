import numpy as np

def binary_crossentropy(Yhat, Y, derivative=False):
    if derivative:
        return - (np.divide(Y, Yhat) - np.divide(1. - Y, 1. - Yhat))
    else:
        return - (Y * np.log(Yhat) + ((1. - Y)*np.log(1. - Yhat)))
