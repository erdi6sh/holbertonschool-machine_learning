#!/usr/bin/env python3
"""
This module contains a function `add_arrays` that adds
two arrays element-wise and returns a new list. Returns None
if the arrays are not the same length.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): First list of numbers (ints or floats)
        arr2 (list): Second list of numbers (ints or floats)

    Returns:
        list: A new list containing the sum of elements of arr1 and arr2
        None: If arr1 and arr2 are not of the same length
    """
    if len(arr1) == len(arr2):
        arr_sum = []
        for i in range(len(arr1)):
            arr_sum.append(arr1[i] + arr2[i])
        return arr_sum
    else:
        return None
