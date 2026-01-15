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
        """
        Calculates the cost using logistic regression

        Args:
            Y: numpy array with shape (1, m) with correct labels
            A: numpy array with shape (1, m) with activated output

        Returns:
            The cost (log loss)
        """
        m = Y.shape[1]
        log_loss = (-1 / m * np.sum(Y * np.log(A) +
                    (1 - Y) * np.log(1.0000001 - A)))
        return log_loss

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy array with shape (nx, m) containing input data
            Y: numpy array with shape (1, m) with correct labels

        Returns:
            Tuple of (predictions, cost)
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy array with shape (nx, m) containing input data
            Y: numpy array with shape (1, m) with correct labels
            A: numpy array with shape (1, m) with activated output
            alpha: learning rate

        Updates:
            __W and __b using gradient descent
        """
        m = Y.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)

        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
