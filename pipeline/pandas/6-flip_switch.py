#!/usr/bin/env python3
"""Module that sorts and transposes a DataFrame."""


def flip_switch(df):
    """
    Sorts a DataFrame in reverse chronological order and transposes it.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The transposed DataFrame sorted in reverse order.
    """
    df = df.sort_values(by="Timestamp", ascending=False)
    return df.T
