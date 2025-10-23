#!/usr/bin/env python3
"""
Module 10-matisse
Calculates the derivative of a polynomial represented by a list of coefficients
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    The polynomial is represented as a list of coefficients.
    The index of each coefficient represents the power of x.

    Example:
        If f(x) = x^3 + 3x + 5, then poly = [5, 3, 0, 1]
        f'(x) = 3x^2 + 3, so the output is [3, 0, 3]

    Args:
        poly (list): List of coefficients representing a polynomial.

    Returns:
        list: A new list of coefficients representing the derivative.
        None: If poly is not valid.
    """
    if (not isinstance(poly, list)) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        coeff = poly[power] * power
        derivative.append(coeff)

    # If derivative is all zeros, return [0]
    if all(c == 0 for c in derivative):
        return [0]

    return derivative
