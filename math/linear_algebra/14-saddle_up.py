#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication
using NumPy.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication between two NumPy arrays.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: A new array representing the product of
        mat1 and mat2.
    """
    return np.matmul(mat1, mat2)
