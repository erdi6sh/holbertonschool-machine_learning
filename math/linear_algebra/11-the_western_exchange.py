#!/usr/bin/env python3
"""
Module that provides a function to transpose a NumPy array.
"""


def np_transpose(matrix):
    """
    Returns the transpose of a NumPy ndarray.

    Args:
        matrix (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: A new array that is the transpose of matrix.
    """
    return matrix.T
