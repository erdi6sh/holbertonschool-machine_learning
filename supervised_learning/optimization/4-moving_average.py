#!/usr/bin/env python3
"""Module for calculating moving average."""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Args:
         list of data to calculate the moving average of
        beta: weight used for the moving average

    Returns:
        A list containing the moving averages of data
    """
    moving_averages = []
    v = 0

    for t, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value
        bias_correction = 1 - beta ** t
        moving_averages.append(v / bias_correction)

    return moving_averages
