import os
import matplotlib.pyplot as plt

def plot_costs(costs):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.savefig(os.getcwd().replace('src', '') + '/data/costs')
    plt.clf()