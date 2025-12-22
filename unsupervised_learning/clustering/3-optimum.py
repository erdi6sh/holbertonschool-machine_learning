#!/usr/bin/env python3
"""Find optimum number of clusters by variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: minimum number of clusters to check for (inclusive)
        kmax: maximum number of clusters to check for (inclusive)
        iterations: maximum number of iterations for K-means

    Returns:
        results: list containing the outputs of K-means for each cluster size
        d_vars: list containing the difference in variance from smallest
                cluster size
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None

        results.append((C, clss))
        var = variance(X, C)
        variances.append(var)

    first_var = variances[0]
    d_vars = [first_var - v for v in variances]

    return results, d_vars
