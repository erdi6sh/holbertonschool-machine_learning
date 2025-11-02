#!/usr/bin/env python3
"""Module that rearranges and concatenates two DataFrames with hierarchy."""


index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two DataFrames within a timestamp range and rearranges index.

    Args:
        df1 (pd.DataFrame): The first DataFrame (coinbase).
        df2 (pd.DataFrame): The second DataFrame (bitstamp).

    Returns:
        pd.DataFrame: The hierarchical concatenated DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    df = df.swaplevel(0, 1).sort_index()
    return df
