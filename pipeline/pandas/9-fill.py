#!/usr/bin/env python3
"""Module that fills missing values and removes Weighted_Price column."""


def fill(df):
    """
    Fills missing values and removes the Weighted_Price column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])
    df["Close"] = df["Close"].fillna(method="ffill")
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)
    return df
