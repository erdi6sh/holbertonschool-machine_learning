#!/usr/bin/env python3
"""Module that creates a DataFrame from a NumPy array"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.
    Columns should be labeled in alphabetical order and capitalized.

    Parameters:
    array (np.ndarray): The NumPy array to convert

    Returns:
    pd.DataFrame: DataFrame with labeled columns
    """
    # Number of columns in the array
    num_cols = array.shape[1]

    # Generate column labels: 'A', 'B', 'C', ...
    columns = [chr(ord('A') + i) for i in range(num_cols)]

    # Create the DataFrame
    df = pd.DataFrame(data=array, columns=columns)

    return df
