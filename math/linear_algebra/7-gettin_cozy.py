#!/usr/bin/env python3
"""
Module that provides a function to concatenate two 2D matrices.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specified axis.

    Args:
        mat1 (list of lists): The first 2D matrix.
        mat2 (list of lists): The second 2D matrix.
        axis (int): The axis along which to concatenate.
                    0 means along rows (vertical),
                    1 means along columns (horizontal).

    Returns:
        list of lists: A new 2D matrix that is the concatenation
        of mat1 and mat2, or None if they cannot be concatenated.
    """
    if not mat1 or not mat2:
        return None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    return None
