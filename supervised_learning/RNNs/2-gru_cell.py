#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell"""

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: numpy.ndarray of shape (m, h), previous hidden state
        x_t: numpy.ndarray of shape (m, i), data input for the cell

        Returns: h_next, y
            h_next: the next hidden state
            y: the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self._sigmoid(concat @ self.Wz + self.bz)

        # Reset gate
        r = self._sigmoid(concat @ self.Wr + self.br)

        # Intermediate hidden state (candidate)
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_candidate = np.tanh(concat_r @ self.Wh + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_candidate

        # Output with softmax
        out = h_next @ self.Wy + self.by
        e = np.exp(out - np.max(out, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)

        return h_next, y

    def _sigmoid(self, x):
        """Applies the sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
