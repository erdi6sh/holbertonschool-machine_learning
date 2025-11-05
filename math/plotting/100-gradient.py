#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def gradient():

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    mountain_elevation = z
    scatter = plt.scatter(x, y, c=mountain_elevation, cmap='viridis', s=30, edgecolor='k', alpha=0.85)
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.title("Mountain Elevation")
    cbar = plt.colorbar(scatter)
    cbar.set_label("elevation (m)")

    plt.show()


gradient()
