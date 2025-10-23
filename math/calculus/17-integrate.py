#!/usr/bin/env python3
"""
Module 17-integrate
Calculates the integral of a polynomial represented by a list of coefficients.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    The polynomial is represented as a list of coefficients.
    The index of each coefficient represents the power of x.

    Example:
        If f(x) = x^3 + 3x + 5, then poly = [5, 3, 0, 1]
        Integral: F(x) = 5x + 1.5x^2 + 0x^3 + 0.25x^4 + C
        Output: [C, 5, 1.5, 0, 0.25]

    Args:
        poly (list): List of coefficients representing a polynomial.
        C (int, optional): Integration constant. Defaults to 0.

    Returns:
        list: New list of coefficients representing the integral.
        None: If poly or C are not valid.
    """
    if (not isinstance(poly, list)) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for power in range(len(poly)):
        coeff = poly[power] / (power + 1)
        # Store as int if it’s a whole number
        if coeff.is_integer():
            coeff = int(coeff)
        integral.append(coeff)

    # Remove unnecessary trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
