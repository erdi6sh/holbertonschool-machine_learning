#!/usr/bin/env python3
"""Binomial distribution class implementation.
This module defines a class that represents a Binomial distribution.
"""


class Binomial:
    """Represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a Binomial distribution.

        Args:
            data (list, optional): A list of the data to be used to estimate
                the distribution parameters. Defaults to None.
            n (int, optional): The number of trials. Defaults to 1.
            p (float, optional): The probability of a success. Defaults to 0.5.

        Raises:
            ValueError: If n is not a positive value.
            ValueError: If p is not greater than 0 and less than 1.
            TypeError: If data is provided and is not a list.
            ValueError: If data contains fewer than two values.
        """
        if data is None:
            # No data provided — use given n and p values
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Data provided — estimate parameters from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance from data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate p and n using mean and variance relationships
            p_first = 1 - (variance / mean)
            n_round = mean / p_first

            self.n = round(n_round)
            self.p = mean / self.n
