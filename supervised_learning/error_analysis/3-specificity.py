#!/usr/bin/env python3
"""Module for calculating specificity."""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)
                   Row indices represent correct labels
                   Column indices represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing specificity
        of each class
    """
    total = np.sum(confusion)
    row_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)
    tp = np.diag(confusion)

    tn = total - row_sums - col_sums + tp
    fp = col_sums - tp

    return tn / (tn + fp)
