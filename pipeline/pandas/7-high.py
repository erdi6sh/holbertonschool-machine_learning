#!/usr/bin/env python3
"""Module that sorts a DataFrame by the High column."""


def high(df):
    """
    Sorts a DataFrame by the High column in descending order.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame sorted by the High column.
    """
    return df.sort_values(by="High", ascending=False)
