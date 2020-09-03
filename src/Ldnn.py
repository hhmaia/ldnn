import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

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


def binary_crossentropy(Yhat, Y, derivative=False):
    if derivative:
        return - (np.divide(Y, Yhat) - np.divide(1. - Y, 1. - Yhat))
    else:
        return - (Y * np.log(Yhat) + ((1. - Y)*np.log(1. - Yhat)))


def compute_cost(AL, Y, loss=binary_crossentropy):
    m = Y.shape[1]

    cost = (1/m) * np.sum(loss(AL, Y))
    cost = cost.squeeze()
    assert cost.shape == ()
    return cost


def initialize_parameters(n_dims, w_scale=0.01, seed=1):
    np.random.seed(seed)
    params = {}
    for l in range(1, len(n_dims)):
        params[l] = {}
        W = np.random.randn(n_dims[l], n_dims[l-1]) / np.sqrt(n_dims[l-1]) #* w_scale
        b = np.zeros((n_dims[l], 1))
        params[l]['W'] = W
        params[l]['b'] = b
    return params


def forward_linear(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def forward_linear_activation(A_prev, W, b, activation=sigmoid):
    assert (W.shape[0] == b.shape[0])
    assert (A_prev.shape[0] == W.shape[1])

    Z, linear_cache = forward_linear(A_prev, W, b)
    A = activation(Z)

    assert (A.shape == Z.shape)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache


def backward_activation(dA, cache, activation_derivative):
    Z = cache
    dZ = dA * activation_derivative(Z)
    return dZ


def backward_linear(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def backward_linear_activation(dA, cache, activation_derivative):
    linear_cache, activation_cache = cache
    dZ = backward_activation(dA, activation_cache, activation_derivative)
    dA_prev, dW, db = backward_linear(dZ, linear_cache)
    return dA_prev, dW, db


def l_model_forward(X,
                    params,
                    hidden_layers_activation=relu,
                    output_layer_activation=sigmoid):
    caches = []
    A = X
    L = len(params)

    for l in range(1, L):
        A_prev = A
        A, cache = forward_linear_activation(A_prev, params[l]['W'], params[l]['b'], hidden_layers_activation)
        caches.append(cache)

    AL, cache = forward_linear_activation(A, params[L]['W'], params[L]['b'], output_layer_activation)
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))
    return AL, caches


def l_model_backward(AL, Y, caches,
                     hidden_layers_activation = relu,
                     output_layer_activation = sigmoid):
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    for l in range(L+1):
        grads[l] = {}

    current_cache = caches[L-1]
    dAL = binary_crossentropy(AL, Y, derivative=True)
    grads[L-1]['dA'], grads[L]['dW'], grads[L]['db'] = \
        backward_linear_activation(dAL, current_cache,
                                   partial(output_layer_activation, derivative=True))

    # For L-2 to 0 (penultimate layer to first layer)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads[l]['dA'], grads[l+1]['dW'], grads[l+1]['db'] = \
        backward_linear_activation(grads[l+1]['dA'], current_cache,
                                   partial(hidden_layers_activation, derivative=True))

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)
    for l in range(1, L+1):
        parameters[l]['W'] -= learning_rate * grads[l]['dW']
        parameters[l]['b'] -= learning_rate * grads[l]['db']

    return parameters


def l_layer_model_train(X, Y,
                  layer_dims,
                  epochs,
                  learning_rate=.0075,
                  hidden_layers_activation=relu,
                  output_layer_activation=sigmoid,
                  weights_init_scale=0.01,
                  loss=binary_crossentropy,
                  print_costs=False,
                  plot_costs=True):

    params = initialize_parameters(layer_dims, w_scale=weights_init_scale)
    costs = []

    for epoch in range(epochs):
        AL, caches = l_model_forward(X, params,
                                     hidden_layers_activation=hidden_layers_activation,
                                     output_layer_activation=output_layer_activation)

        cost = compute_cost(AL, Y, loss)
        grads = l_model_backward(AL, Y, caches,
                                 hidden_layers_activation=hidden_layers_activation,
                                 output_layer_activation=output_layer_activation)

        params = update_parameters(params, grads, learning_rate)
        costs.append(cost)
        # Print the cost every 100 training example
        if print_costs and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" %(epoch, cost))

    # plot the cost
    if plot_costs:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig(os.getcwd().replace('src', '') + '/data/costs')
        plt.clf()

    return params
