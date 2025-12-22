#!/usr/bin/env python3
"""Expectation Maximization for Gaussian Mixture Model"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a GMM

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing max iterations
        tol: non-negative float containing tolerance for early stopping
        verbose: boolean that determines if info should be printed

    Returns:
        pi: numpy.ndarray of shape (k,) containing priors
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariances
        g: numpy.ndarray of shape (k, n) containing probabilities
        l: log likelihood of the model
        or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape
    if k > n:
        return None, None, None, None, None

    try:
        pi, m, S = initialize(X, k)
        if pi is None:
            return None, None, None, None, None

        l_prev = 0

        for i in range(iterations):
            g, l = expectation(X, pi, m, S)
            if g is None:
                return None, None, None, None, None

            if verbose:
                if i == 0 or (i + 1) % 10 == 0:
                    print(f"Log Likelihood after {i} iterations: {l:.5f}")

            if i > 0 and abs(l - l_prev) <= tol:
                if verbose:
                    print(f"Log Likelihood after {i} iterations: {l:.5f}")
                break

            l_prev = l

            pi, m, S = maximization(X, g)
            if pi is None:
                return None, None, None, None, None

        if verbose and i == iterations - 1:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

        return pi, m, S, g, l

    except Exception:
        return None, None, None, None, None
