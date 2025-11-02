#!/usr/bin/env python3
"""Module that slices a DataFrame to select specific columns and rows."""

import pandas as pd


def slice(df):
    """
    Extracts specific columns and selects every 60th row from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing cryptocurrency data.

    Returns:
        pd.DataFrame: A sliced DataFrame with columns 'High', 'Low',
        'Close', and 'Volume_(BTC)', keeping every 60th row.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]
