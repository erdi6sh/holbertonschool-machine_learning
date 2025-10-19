#!/usr/bin/env python3
"""
Module that provides a function to concatenate two NumPy arrays.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two NumPy arrays along a specified axis.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray): The second array.
        axis (int): The axis along which to concatenate the arrays.

    Returns:
        numpy.ndarray: A new array that is the concatenation of
        mat1 and mat2 along the given axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
