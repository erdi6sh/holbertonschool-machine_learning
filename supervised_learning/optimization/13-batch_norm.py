#!/usr/bin/env python3
"""Module for batch normalization."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output using batch normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) that should be normalized
           m is the number of data points
           n is the number of features in Z
        gamma: numpy.ndarray of shape (1, n) containing scales
        beta: numpy.ndarray of shape (1, n) containing offsets
        epsilon: small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_out = gamma * Z_norm + beta
    return Z_out
