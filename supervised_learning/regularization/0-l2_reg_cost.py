#!/usr/bin/env python3
"""Module for calculating L2 regularization cost."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of weights and biases (numpy.ndarrays)
        L: number of layers in the neural network
        m: number of data points used

    Returns:
        Cost of the network accounting for L2 regularization
    """
    l2_sum = 0

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        l2_sum += np.linalg.norm(W) ** 2

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum

    return l2_cost
