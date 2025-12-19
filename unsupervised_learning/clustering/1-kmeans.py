#!/usr/bin/env python3
"""K-means clustering algorithm"""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means"""
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


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing max iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster index
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    for i in range(iterations):
        C_old = C.copy()

        distances = np.sqrt(((X[:, np.newaxis] - C) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            cluster_points = X[clss == j]

            if len(cluster_points) == 0:
                C[j] = np.random.uniform(low=min_vals, high=max_vals,
                                         size=(d,))
            else:
                C[j] = cluster_points.mean(axis=0)

        if np.allclose(C, C_old):
            break

    distances = np.sqrt(((X[:, np.newaxis] - C) ** 2).sum(axis=2))
    clss = np.argmin(distances, axis=1)

    return C, clss
