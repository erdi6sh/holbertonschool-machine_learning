#!/usr/bin/env python3
"""Module for gradient descent with dropout."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights using gradient descent with Dropout regularization.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct
           labels
        weights: dictionary of weights and biases of the network
        cache: dictionary of outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network

    The weights are updated in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            dA = np.matmul(W.T, dZ)
            dA = dA * cache['D' + str(layer - 1)]
            dA = dA / keep_prob
            dZ = dA * (1 - A_prev ** 2)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
