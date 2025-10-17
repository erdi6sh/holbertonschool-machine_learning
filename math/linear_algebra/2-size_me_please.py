#!/usr/bin/env python3
"""
This module contains a function `matrix_shape` that calculates
the shape of a matrix (nested lists) and returns it as a list of integers.
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix (list of lists).
    Returns a list of integers representing the size of each dimension.
    """
    shape = []

    # Step 2: create an infinite loop (we'll break manually)
    for _ in iter(int, 1):
        # Step 3: check if current level is a list
        if isinstance(matrix, list):
            # Step 4: append number of elements at this level
            shape.append(len(matrix))
            # Step 5: go one level deeper (first element)
            matrix = matrix[0]
        else:
            # Step 6: reached a non-list, stop looping
            break
    return shape
