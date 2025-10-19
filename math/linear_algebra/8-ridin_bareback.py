#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of lists of int/float): The first matrix.
        mat2 (list of lists of int/float): The second matrix.

    Returns:
        list of lists of int/float: A new matrix representing the
        product of mat1 and mat2.
        Returns None if the matrices cannot be multiplied.
    """
    # Check if matrices can be multiplied
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
