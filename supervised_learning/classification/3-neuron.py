#!/usr/bin/env python3
"""Neuron class for binary classification"""
import numpy as np


class Neuron:
    """Single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize neuron with nx input features"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neuron

        Args:
            X: numpy array with shape (nx, m) containing input data

        Returns:
            The activated output (self.__A) using sigmoid activation
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):

        m = Y.shape[1]

        log_loss = -1/m*np.sum(Y * np.log(A) + (1-Y)*(np.log(1.0000001-A)))

        return log_loss
