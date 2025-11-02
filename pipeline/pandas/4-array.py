#!/usr/bin/env python3
"""Module that converts last 10 rows of High and Close columns to ndarray."""


def array(df):
    """
    Selects the last 10 rows of 'High' and 'Close' columns from a DataFrame
    and converts them to a NumPy ndarray.

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Close' columns.

    Returns:
        numpy.ndarray: Array containing the last 10 rows of High and Close.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
