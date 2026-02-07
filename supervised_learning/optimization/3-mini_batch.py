#!/usr/bin/env python3
"""Module for creating mini-batches."""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network.

    Args:
        X: numpy.ndarray of shape (m, nx) representing input data
           m is the number of data points
           nx is the number of features in X
        Y: numpy.ndarray of shape (m, ny) representing the labels
           m is the same number of data points as in X
           ny is the number of classes
        batch_size: number of data points in a batch

    Returns:
        List of mini-batches containing tuples (X_batch, Y_batch)
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    num_complete_batches = m // batch_size

    for i in range(num_complete_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        start = num_complete_batches * batch_size
        X_batch = X_shuffled[start:]
        Y_batch = Y_shuffled[start:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
