#!/usr/bin/env python3
"""Module for forward propagation with dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing input data
        weights: dictionary of weights and biases of the network
        L: number of layers in the network
        keep_prob: probability that a node will be kept

    Returns:
        Dictionary containing outputs of each layer and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]

        Z = np.matmul(W, A_prev) + b

        if layer == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = A * D
            A = A / keep_prob
            cache['D' + str(layer)] = D

        cache['A' + str(layer)] = A

    return cache
