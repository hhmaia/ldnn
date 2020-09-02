import os
import numpy as np
import matplotlib.pyplot as plt


def create_batches_generator(x, y, batch_size, shuffle=True):
    assert x.shape[1] == y.shape[1]
    m = x.shape[1]
    if shuffle:
        permutations = np.random.permutation(m)
        x = x[:, permutations]
        y = y[:, permutations]

    for batch_i in range(0, m, batch_size):
        yield x[:, batch_i:batch_i+batch_size], y[:, batch_i:batch_i+batch_size]


def plot_costs(costs):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.savefig(os.getcwd().replace('src', '') + '/data/costs')
    plt.clf()

