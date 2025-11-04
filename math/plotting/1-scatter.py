#!/usr/bin/env python3
"""This module generates a scatter plot of Men's Height vs Weight."""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """Plots Men's Height vs Weight as a scatter plot."""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plotting data as magenta points
    plt.scatter(x, y, color='magenta')

    # Adding labels and title
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")

    # Display the plot
    plt.show()
