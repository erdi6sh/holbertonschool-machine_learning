#!/usr/bin/env python3
"""Module that computes descriptive statistics of a DataFrame."""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics.
    """
    return df.drop(columns=["Timestamp"]).describe()
