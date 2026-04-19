#!/usr/bin/env python3
"""LSTM Cell"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: numpy.ndarray of shape (m, h), previous hidden state
        c_prev: numpy.ndarray of shape (m, h), previous cell state
        x_t: numpy.ndarray of shape (m, i), data input for the cell

        Returns: h_next, c_next, y
            h_next: the next hidden state
            c_next: the next cell state
            y: the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = self._sigmoid(concat @ self.Wf + self.bf)

        # Update (input) gate
        u = self._sigmoid(concat @ self.Wu + self.bu)

        # Candidate cell state
        c_candidate = np.tanh(concat @ self.Wc + self.bc)

        # Next cell state
        c_next = f * c_prev + u * c_candidate

        # Output gate
        o = self._sigmoid(concat @ self.Wo + self.bo)

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output with softmax
        out = h_next @ self.Wy + self.by
        e = np.exp(out - np.max(out, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)

        return h_next, c_next, y

    def _sigmoid(self, x):
        """Applies the sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
