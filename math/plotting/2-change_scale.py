#!/usr/bin/env python3
"""This module generates a plot for the exponential decay of C-14."""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Plots the exponential decay of C-14 with a logarithmic y-axis."""
    x = np.arange(0, 28651, 5730)  # Time in years
    r = np.log(0.5)  # Decay constant
    t = 5730  # Half-life of C-14 in years
    y = np.exp((r / t) * x)  # Exponential decay equation

    plt.figure(figsize=(6.4, 4.8))

    # Plot the data
    plt.plot(x, y)

    # Set labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    # Set logarithmic scale for y-axis
    plt.yscale('log')

    # Set x-axis limits to range from 0 to 28650
    plt.xlim(0, 28650)

    # Display the plot
    plt.show()
