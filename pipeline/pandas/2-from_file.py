#!/usr/bin/env python3
"""
Module that loads a file as a pandas DataFrame.

Contains the function:
- from_file(filename, delimiter)
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file into a pandas DataFrame.

    Parameters:
        filename (str): Path to the file to load
        delimiter (str): Column separator (e.g., ',' for CSV)

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(filename, sep=delimiter)
    return df
