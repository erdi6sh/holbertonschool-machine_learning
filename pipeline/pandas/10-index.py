#!/usr/bin/env python3
"""Module that sets the Timestamp column as the DataFrame index."""

import pandas as pd


def index(df):
    """
    Sets the Timestamp column as the DataFrame index.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with Timestamp as index.
    """
    return df.set_index("Timestamp")
