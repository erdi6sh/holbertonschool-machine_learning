#!/usr/bin/env python3
"""
Module that plots a histogram of student grades.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Creates a histogram showing the distribution of
    student grades for a project.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.show()
