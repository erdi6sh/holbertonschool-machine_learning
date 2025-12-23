#!/usr/bin/env python3
"""Expectation step in EM algorithm for GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        pi: numpy.ndarray of shape (k,) containing priors for each cluster
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices

    Returns:
        g: numpy.ndarray of shape (k, n) containing posterior probabilities
        l: total log likelihood
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        likelihoods = np.zeros((k, n))

        for i in range(k):
            P = pdf(X, m[i], S[i])
            if P is None:
                return None, None
            likelihoods[i] = pi[i] * P

        marginal = np.sum(likelihoods, axis=0)
        g = likelihoods / marginal
        l = np.sum(np.log(marginal))  # noqa: E741

        return g, l
    except Exception:
        return None, None
