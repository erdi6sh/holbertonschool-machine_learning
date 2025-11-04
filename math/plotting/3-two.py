#!/usr/bin/env python3
"""This module plots the exponential decay of two radioactive elements."""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """Plots the exponential decay of C-14 and Ra-226."""

    # Time in years (from 0 to 20,000)
    x = np.arange(0, 20001, 1000)

    # Decay constant
    r = np.log(0.5)

    # Half-life of C-14
    t1 = 5730

    # Half-life of Ra-226
    t2 = 1600

    # Exponential decay for C-14
    y1 = np.exp((r / t1) * x)

    # Exponential decay for Ra-226
    y2 = np.exp((r / t2) * x)

    plt.figure(figsize=(6.4, 4.8))

    # Plotting the two lines with specified styles
    plt.plot(x, y1, 'r--', label='C-14')  # Dashed red line for C-14
    plt.plot(x, y2, 'g-', label='Ra-226')  # Solid green line for Ra-226

    # Set labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    # Set axis limits
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Add legend in the upper right corner
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()
