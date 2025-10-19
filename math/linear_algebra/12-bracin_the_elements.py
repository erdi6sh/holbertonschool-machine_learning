#!/usr/bin/env python3
"""
Module that provides a function to perform element-wise
operations on NumPy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division between two NumPy arrays.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray or scalar): The second array or a scalar.

    Returns:
        tuple: A tuple containing four NumPy arrays:
            - The element-wise sum
            - The element-wise difference
            - The element-wise product
            - The element-wise quotient
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
