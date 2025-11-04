#!/usr/bin/env python3
"""
Plot y = x^3 as a solid red line with x-axis from 0 to 10.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot a cubic curve y = x^3 for x in 0..10 as a solid red line."""
    x = np.arange(0, 11)
    y = x ** 3

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, 'r-', linewidth=2)  # solid red line
    plt.xlim(0, 10)  # x-axis exactly 0 to 10
    plt.show()
