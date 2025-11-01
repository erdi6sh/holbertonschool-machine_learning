# Pandas Project - Holberton School

## Overview
This project explores the use of **pandas** for data manipulation and analysis in Python.  
You will work with real-world datasets from **Coinbase** and **Bitstamp**, and perform tasks such as creating DataFrames, slicing, indexing, merging, and analyzing data.

---

## Learning Objectives
At the end of this project, you will be able to:

- Explain what **pandas** is and why it is used.
- Create and manipulate **pd.DataFrame** and **pd.Series** objects.
- Load data from CSV files into DataFrames.
- Perform indexing and slicing on DataFrames.
- Use hierarchical indexing.
- Reassign and rename columns.
- Sort and filter data using boolean logic.
- Merge, concatenate, and join DataFrames.
- Extract statistical information from datasets.
- Visualize data from DataFrames.

---

## Datasets
We are using the following datasets for this project:

1. **Bitstamp**: `bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv`
2. **Coinbase**: `coinbaseUSD_1-min_data_2014-12-01_to_2020-04-22.csv`

These datasets contain historical cryptocurrency prices (Open, High, Low, Close, Volume).

---

## Files

| Filename | Description |
|----------|-------------|
| `0-from_numpy.py` | Contains the function `from_numpy(array)` that converts a NumPy array into a DataFrame with alphabetical columns. |
| `0-main.py` | Test file for `from_numpy.py`. Demonstrates conversion of NumPy arrays into DataFrames. |
| `README.md` | This file. Provides an overview of the project and instructions. |

---

## Installation

Ensure you have **Python 3.9**, **NumPy 1.25.2**, and **pandas 2.2.2** installed.  

To install pandas:

```bash
pip install --user pandas==2.2.2
