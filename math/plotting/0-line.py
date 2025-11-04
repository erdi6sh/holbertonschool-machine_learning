#!/usr/bin/env python3
"""
Module that plots a simple line graph.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots y = x^3 as a solid red line
    for x values from 0 to 10.
    """
    x = np.arange(0, 11)
    y = x ** 3

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, color='red', linestyle='-', linewidth=1)
    plt.xlim(0, 10)
    plt.show()
