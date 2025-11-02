#!/usr/bin/env python3
"""Module that renames and formats a DataFrame timestamp column."""

import pandas as pd


def rename(df):
    """
    Rename the 'Timestamp' column to 'Datetime', convert its values to
    pandas datetime objects (timestamps are in seconds), keep only the
    'Datetime' and 'Close' columns and return the modified DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: Modified DataFrame with 'Datetime' and 'Close'.
    """
    # Rename column in-place
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

    # Convert the renamed column to datetime (timestamps are in seconds)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Keep only the requested columns
    df = df[['Datetime', 'Close']]

    return df
