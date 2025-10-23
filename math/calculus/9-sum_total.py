#!/usr/bin/env python3
"""
Module 9-sum_total
Calculates the summation of i² from i = 1 to n without using loops.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1² to n².

    Formula:
        sum(i²) = n(n + 1)(2n + 1) / 6

    Args:
        n (int): The upper limit of the summation (must be a positive integer).

    Returns:
        int: The result of the summation.
        None: If n is not a valid positive integer.
    """
    if type(n) is not int or n < 1:
        return None

    return int(n * (n + 1) * (2 * n + 1) / 6)
