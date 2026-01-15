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

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
