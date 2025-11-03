#!/usr/bin/env python3
"""Transform and visualize daily aggregated price data from a DataFrame."""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


def _prepare(df):
    """
    Internal helper to transform the minute-level dataframe into a
    daily-aggregated dataframe according to project specs.

    Steps:
    - Remove Weighted_Price column
    - Rename Timestamp -> Date and convert to midnight datetime
    - Set Date as index
    - Fill missing values as required
    - Filter to 2017-01-01 and later
    - Resample daily with requested aggregations
    """
    # remove Weighted_Price column if present
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # rename Timestamp to Date and convert to datetime (normalize to midnight)
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], unit="s").dt.normalize()

    # set index to Date
    df = df.set_index("Date")

    # Fill missing Close with previous value
    df["Close"] = df["Close"].fillna(method="ffill")

    # Fill High/Low/Open missing values with the same row's Close
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    # Fill volumes missing with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    # select 2017 and beyond
    df = df.loc[df.index >= pd.to_datetime("2017-01-01")]

    # resample daily and aggregate as requested
    agg = {
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    }
    df_daily = df.resample("D").agg(agg)

    return df_daily


if __name__ == "__main__":
    df = from_file("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ",")
    df_daily = _prepare(df)

    # Return/print transformed DataFrame before plotting (script prints it)
    print(df_daily)

    # Plot daily aggregated data (opens a window or inline in notebook)
    ax = df_daily.plot(
        title="Daily aggregated coinbase Weighted metrics (2017+)",
        ylabel="Price / Volume",
        figsize=(12, 6),
    )
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.show()
