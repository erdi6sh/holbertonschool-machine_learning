#!/usr/bin/env python3
"""Calculate total intra-cluster variance"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing the centroid means

    Returns:
        var: total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    try:
        distances_squared = ((X[:, np.newaxis] - C) ** 2).sum(axis=2)
        min_distances_squared = distances_squared.min(axis=1)
        var = min_distances_squared.sum()
        return var
    except Exception:
        return None
