#!/usr/bin/env python3
"""Module that removes rows with NaN values in the Close column."""

import pandas as pd


def prune(df):
    """
    Removes rows with NaN values in the Close column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame without NaN values in Close.
    """
    return df.dropna(subset=["Close"])
