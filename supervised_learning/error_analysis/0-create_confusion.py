#!/usr/bin/env python3
"""Module for creating confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
                with correct labels
                m is the number of data points
                classes is the number of classes
        logits: one-hot numpy.ndarray of shape (m, classes)
                with predicted labels

    Returns:
        Confusion numpy.ndarray of shape (classes, classes)
        Row indices represent correct labels
        Column indices represent predicted labels
    """
    confusion = np.matmul(labels.T, logits)
    return confusion
