#!/usr/bin/env python3
"""
Module that creates a pandas DataFrame from a dictionary.

The DataFrame has:
- Columns: 'First' with numeric values, 'Second' with strings
- Row labels: 'A', 'B', 'C', 'D'
The DataFrame is saved into the variable df.
"""

import pandas as pd

df = pd.DataFrame(
    {
        'First': [0.0, 0.5, 1.0, 1.5],
        'Second': ['one', 'two', 'three', 'four']
    },
    index=['A', 'B', 'C', 'D']
)
