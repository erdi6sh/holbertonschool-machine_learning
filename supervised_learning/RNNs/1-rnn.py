#!/usr/bin/env python3
"""RNN Forward Propagation"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    rnn_cell: instance of RNNCell used for forward propagation
    X: numpy.ndarray of shape (t, m, i), data to be used
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    h_0: numpy.ndarray of shape (m, h), initial hidden state
        h: dimensionality of the hidden state

    Returns: H, Y
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
