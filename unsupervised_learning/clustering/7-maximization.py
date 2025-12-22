#!/usr/bin/env python3
"""Maximization step in EM algorithm for GMM"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
        pi: numpy.ndarray of shape (k,) containing updated priors
        m: numpy.ndarray of shape (k, d) containing updated means
        S: numpy.ndarray of shape (k, d, d) containing updated covariances
        or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    try:
        N_k = np.sum(g, axis=1)
        pi = N_k / n
        m = (g @ X) / N_k[:, np.newaxis]

        S = np.zeros((k, d, d))

        for i in range(k):
            X_centered = X - m[i]
            weighted = g[i, :, np.newaxis] * X_centered
            S[i] = (weighted.T @ X_centered) / N_k[i]

        return pi, m, S
    except Exception:
        return None, None, None
