#!/usr/bin/env python3
"""This module contains a function that adds two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise and returns a new matrix.
    If the matrices are not the same shape, returns None.
    """
    # Check that both matrices have the same number of rows
    if len(mat1) != len(mat2):
        return None

    added_matrices = []

    # Loop through each row
    for i in range(len(mat1)):
        # Check that each corresponding row has the same length
        if len(mat1[i]) != len(mat2[i]):
            return None

        row_sum = []
        # Loop through each column element
        for j in range(len(mat1[0])):
            row_sum.append(mat1[i][j] + mat2[i][j])
        added_matrices.append(row_sum)

    return added_matrices
