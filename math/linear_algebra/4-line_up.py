#!/usr/bin/env python3
"""This code adds 2 arrays element-wise """


def add_arrays(arr1, arr2):
    if len(arr1) == len(arr2):
        arr_sum = []
        for i in range(len(arr1)):
            arr_sum.append(arr1[i] + arr2[i])
        return arr_sum
    else:
        return None
