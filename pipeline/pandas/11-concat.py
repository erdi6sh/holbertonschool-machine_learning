#!/usr/bin/env python3
"""Module that concatenates two DataFrames with labeled keys."""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two DataFrames after indexing on Timestamp.

    Args:
        df1 (pd.DataFrame): The first DataFrame (coinbase).
        df2 (pd.DataFrame): The second DataFrame (bitstamp).

    Returns:
        pd.DataFrame: The concatenated DataFrame with keys.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    return df
