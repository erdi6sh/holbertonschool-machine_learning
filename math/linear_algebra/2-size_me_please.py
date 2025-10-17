#!/usr/bin/env python3
""" """


def matrix_shape(matrix):
    # Step 1: prepare a list to store the shape
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
