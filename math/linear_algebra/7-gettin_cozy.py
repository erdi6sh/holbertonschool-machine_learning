#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along a given axis"""


# check if either matrix is empty
    if not mat1 or not mat2:
        return None

    # Concatenate along rows
    if axis == 0:
        # Check same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Return a new matrix (deep copy)
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # Concatenate along columns
    elif axis == 1:
        # Check same number of rows
        if len(mat1) != len(mat2):
            return None
        # Merge each row
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    else:
        return None
