import time
from typing import Callable, Any

import numpy as np
from functools import partial

from src.activations import sigmoid, relu
from src.initializers import xavier_init
from src.ldnn_utils import create_batches_generator
from src.losses import binary_crossentropy


def compute_cost(AL, Y, params, l2_lambda, loss=binary_crossentropy):
    """

    :param AL: output matrix of the model
    :param Y: input labels
    :param params: dictionary containing W and b matrices
    :param l2_lambda: l2 regularization parameter
    :param loss: function to be used on loss calculation
    :return: float cost, associated with the current outputs of the network
    """

    # number of examples
    m = Y.shape[1]
    # computation of Frobenius norm for W
    l2_reg_term = 0
    for l in params:
        l2_reg_term += np.square(params[l]['W']).sum()

    # average of the loss plus l2 regularization term. l2_lambda == 0 means
    # no regularization
    cost = (1 / m) * np.sum(loss(AL, Y)) + ((l2_lambda / (2 * m)) * l2_reg_term)
    cost = cost.squeeze()
    assert cost.shape == ()
    return cost


def initialize_parameters(n_dims: Any,
                          weights_initializer: Callable[[int, int], Any],
                          seed=1):
    """
    :param n_dims: number of units in each layer of the network, including
     the input layer.
    :param weights_initializer: function used to create and initialize the
     weights matrix
    :param seed: seed for the random number generation
    :return: dictionary containing L layers, where L is equal to len(n_dims) - 1.
     Each layer is also a dictionary containing the keys 'W' and 'b', which are
     the weights and bias matrix.

     Ex:
        >>> params = initialize_parameters([2, 2, 1], xavier_init)
        >>> params.keys()
        dict_keys([1, 2])
        >>> params[1].keys()
        dict_keys(['W', 'b'])
    """

    np.random.seed(seed)
    params = {}
    for l in range(1, len(n_dims)):
        params[l] = {}
        W = weights_initializer(n_dims[l], n_dims[l - 1])
        b = np.zeros((n_dims[l], 1))
        params[l]['W'] = W
        params[l]['b'] = b
    return params


def forward_linear(A, W, b):
    """

    :param A: inputs for the current layer, can be X (network input)
     or the activation from the previous layer.
    :param W: weights matrix for the layer
    :param b: bias matrix for the layer
    :return Z: output for the layer (before the activation)
    :return cache: A, W and b inputs to be used on other computations
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def forward_linear_activation(A_prev, W, b, activation):
    assert (W.shape[0] == b.shape[0])
    assert (A_prev.shape[0] == W.shape[1])

    # linear_cache is (A_prev, W, b)
    Z, linear_cache = forward_linear(A_prev, W, b)
    A = activation(Z)

    assert (A.shape == Z.shape)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    activation_cache = Z
    # cache is (Z, A_prev, W, b)
    cache = (linear_cache, activation_cache)
    return A, cache


def backward_activation(dA, cache, activation_derivative):
    Z = cache
    dZ = dA * activation_derivative(Z)
    return dZ


def backward_linear(dZ, linear_cache, l2_lambda):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = 1 / m * (np.dot(dZ, A_prev.T) + (l2_lambda * W))
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def backward_linear_activation(dA, cache, activation_derivative, l2_lambda):
    linear_cache, activation_cache = cache
    dZ = backward_activation(dA, activation_cache, activation_derivative)
    dA_prev, dW, db = backward_linear(dZ, linear_cache, l2_lambda)
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
        A, cache = forward_linear_activation(
            A_prev,
            params[l]['W'],
            params[l]['b'],
            hidden_layers_activation)
        caches.append(cache)

    AL, cache = forward_linear_activation(
        A,
        params[L]['W'],
        params[L]['b'],
        output_layer_activation)
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def l_model_backward(AL, Y, caches, l2_lambda,
                     hidden_layers_activation,
                     output_layer_activation):
    L = len(caches)
    grads = {}
    for layer in range(L + 1):
        grads[layer] = {}

    # output layer pass
    current_cache = caches[L - 1]
    dAL = binary_crossentropy(AL, Y, derivative=True)
    ol_activation_d = partial(output_layer_activation, derivative=True)
    grads[L - 1]['dA'], grads[L]['dW'], grads[L]['db'] = \
        backward_linear_activation(
            dAL,
            current_cache,
            ol_activation_d,
            l2_lambda)

    # For L-2 to 0 (penultimate layer to first layer)
    hl_activation_d = partial(hidden_layers_activation, derivative=True)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads[l]['dA'], grads[l + 1]['dW'], grads[l + 1]['db'] = \
            backward_linear_activation(
                grads[l + 1]['dA'],
                current_cache,
                hl_activation_d,
                l2_lambda)

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)
    for l in range(1, L + 1):
        parameters[l]['W'] -= learning_rate * grads[l]['dW']
        parameters[l]['b'] -= learning_rate * grads[l]['db']

    return parameters


def optimization_single_pass(x_batch, y_batch, params,
                             learning_rate,
                             l2_lambda,
                             loss,
                             hl_activation,
                             ol_activation):
    batch_start_time = time.time()
    AL, caches = l_model_forward(x_batch, params, hl_activation, ol_activation)
    grads = l_model_backward(AL, y_batch, caches, l2_lambda, hl_activation,
                             ol_activation)
    cost = compute_cost(AL, y_batch, params, l2_lambda, loss)
    params = update_parameters(params, grads, learning_rate)
    batch_end_time = time.time()
    running_time = batch_end_time - batch_start_time

    return params, cost, running_time


def l_layer_model_train(X, Y,
                        layer_dims,
                        epochs,
                        batch_size=32,
                        learning_rate=.0075,
                        hidden_layers_activation=relu,
                        output_layer_activation=sigmoid,
                        weights_initializer=xavier_init,
                        loss=binary_crossentropy,
                        l2_lambda=0.0,
                        print_costs=False):
    """

    :param X:
    :param Y:
    :param layer_dims:
    :param epochs:
    :param batch_size:
    :param learning_rate:
    :param hidden_layers_activation:
    :param output_layer_activation:
    :param weights_initializer:
    :param loss:
    :param l2_lambda:
    :param print_costs:
    :return:
    """
    assert X.shape[1] == Y.shape[1]

    params = initialize_parameters(layer_dims, weights_initializer)
    costs = []

    # shuffle and partition X and Y into batches
    batches = list(create_batches_generator(X, Y, batch_size))

    for epoch in range(epochs):
        batch_costs = []
        batch_times = []
        for x_batch, y_batch in batches:
            params, cost, rt = optimization_single_pass(
                x_batch,
                y_batch,
                params,
                learning_rate,
                l2_lambda,
                loss,
                hidden_layers_activation,
                output_layer_activation)
            batch_costs.append(cost)
            batch_times.append(rt)

        epoch_cost = np.mean(batch_costs).squeeze()
        epoch_time = np.sum(batch_times).astype(float)
        batch_mean_time = np.mean(batch_times).astype(float)
        costs.append(epoch_cost)
        # Print the cost every 100 training example
        if print_costs and epoch % 100 == 0:
            print("Cost after epoch %i: %f - Batch time: %f, Epoch time: %f" %
                  (epoch, epoch_cost, batch_mean_time, epoch_time))

    return params, costs
