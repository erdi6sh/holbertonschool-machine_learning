#!/usr/bin/env python3
"""K-means clustering initialization"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    if k > n:
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
