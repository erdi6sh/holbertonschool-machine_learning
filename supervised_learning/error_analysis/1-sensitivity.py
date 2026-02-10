#!/usr/bin/env python3
"""Module for calculating sensitivity."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
                   Row indices represent correct labels
                   Column indices represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing sensitivity
        of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
