#!/usr/bin/env python3
"""Module to convert last 10 rows of selected columns to a NumPy array."""

import pandas as pd
import numpy as np


def array(df):
    """
    Selects the last 10 rows of 'High' and 'Close' columns from a DataFrame
    and converts them to a NumPy array.

    Args:
        df (pd.DataFrame): The DataFrame containing 'High' and 'Close' columns.

    Returns:
        np.ndarray: A NumPy array with the last 10 rows of 'High' and 'Close'.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
