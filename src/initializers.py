import numpy as np


def xavier_init(rows, columns):
    """
    Xavier initialization for weights matrix

    :param rows: number of rows for the weights matrix
    :param columns: number of columns
    :return: an uniformly distributed random matrix of shape (rows, columns) scaled to 1/np.sqrt(2/columns)
    """

    return np.random.randn(rows, columns) * np.sqrt(2/columns)