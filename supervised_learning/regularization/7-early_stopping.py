#!/usr/bin/env python3
"""Module for early stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should stop early.

    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the network
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met

    Returns:
        Boolean of whether network should be stopped early,
        followed by updated count
    """
    if opt_cost - cost > threshold:
        return (False, 0)
    else:
        count += 1
        if count >= patience:
            return (True, count)
        return (False, count)
