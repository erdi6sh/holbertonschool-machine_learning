#!/usr/bin/env python3
"""Deep RNN Forward Propagation"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    rnn_cells: list of RNNCell instances of length l
    X: numpy.ndarray of shape (t, m, i), data to be used
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    h_0: numpy.ndarray of shape (l, m, h), initial hidden states
        l: number of layers
        h: dimensionality of the hidden state

    Returns: H, Y
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = []

    for step in range(t):
        x_input = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_input)
            H[step + 1, layer] = h_next
            x_input = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
