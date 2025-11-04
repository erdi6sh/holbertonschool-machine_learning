#!/usr/bin/env python3
"""
Module that combines all five previous plots into one figure.
"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Plots all previous five graphs into one figure arranged in a 3x2 grid.
    """
    # Line plot data
    y0 = np.arange(0, 11) ** 3

    # Scatter plot data
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # Log-scale plot data
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # Two decay plots data
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # Histogram data
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create figure and layout
    plt.figure(figsize=(10, 8))
    plt.suptitle('All in One')

    # Plot 1: Line graph
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1.plot(np.arange(0, 11), y0, color='red')
    ax1.set_xlim(0, 10)
    ax1.set_title('', fontsize='x-small')

    # Plot 2: Scatter graph
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2.scatter(x1, y1, c='magenta', s=10)
    ax2.set_xlabel('Height (in)', fontsize='x-small')
    ax2.set_ylabel('Weight (lbs)', fontsize='x-small')
    ax2.set_title("Men's Height vs Weight", fontsize='x-small')

    # Plot 3: Logarithmic decay
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax3.plot(x2, y2)
    ax3.set_yscale('log')
    ax3.set_xlim(0, 28650)
    ax3.set_xlabel('Time (years)', fontsize='x-small')
    ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax3.set_title('Exponential Decay of C-14', fontsize='x-small')

    # Plot 4: Two decay comparison
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax4.plot(x3, y31, '--', color='red', label='C-14')
    ax4.plot(x3, y32, '-', color='green', label='Ra-226')
    ax4.set_xlim(0, 20000)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize='x-small', loc='upper right')
    ax4.set_title('Exponential Decay of Radioactive Elements',
                  fontsize='x-small')

    # Plot 5: Histogram (bottom row, spans both columns)
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    bins = np.arange(0, 101, 10)
    ax5.hist(student_grades, bins=bins, edgecolor='black')
    ax5.set_xlabel('Grades', fontsize='x-small')
    ax5.set_ylabel('Number of Students', fontsize='x-small')
    ax5.set_title('Project A', fontsize='x-small')

    # Adjust layout for neatness
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
