#!/usr/bin/env python3
"""
This module contains a function `matrix_transpose` that returns
the transpose of a 2D matrix. The function creates a new matrix
with rows and columns swapped.
"""


def matrix_transpose(matrix):
    """
    Returns a new 2D matrix that is the transpose of the input matrix.
    """
    m = len(matrix)        # number of rows
    n = len(matrix[0])     # number of columns

    # Step 2: create a new matrix with n rows and m columns
    new_matrix = [[0 for _ in range(m)] for _ in range(n)]

    # Step 3: fill the new matrix with transposed elements
    for i in range(m):          # iterate over rows of original matrix
        for j in range(n):      # iterate over columns of original matrix
            new_matrix[j][i] = matrix[i][j]
# place element in transposed position
    # Step 4: return the new transposed matrix
    return new_matrix
