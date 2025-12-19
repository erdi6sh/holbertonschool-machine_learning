# K-means Clustering – Initialization

This directory contains the implementation of the **K-means centroid initialization** function used in unsupervised learning.

## File

- `0-initialize.py`: Implements the function `initialize(X, k)` that creates initial centroids for K-means clustering [conversation_history:1].

## Function: `initialize(X, k)`

Initializes cluster centroids for the K-means algorithm using a **multivariate uniform distribution** over the range of the data [conversation_history:1].

### Arguments

- `X`: `numpy.ndarray` of shape `(n, d)`
  - `n`: number of data points  
  - `d`: number of dimensions/features for each data point  
- `k`: positive `int`
  - number of clusters to form [conversation_history:1]

### Behavior

- Computes the **minimum** and **maximum** value of `X` along each dimension (feature) [conversation_history:1].
- Samples `k` points from a **uniform distribution** between these min and max values for every dimension [conversation_history:1].
- Uses `numpy.random.uniform` exactly once to generate all centroids without loops [conversation_history:1].

### Returns

- A `numpy.ndarray` of shape `(k, d)` containing the initialized centroids  
- `None` on failure (invalid inputs or shapes) [conversation_history:1]

### Input Validation

The function returns `None` if:

- `X` is not a 2D `numpy.ndarray`
- `k` is not a positive integer
- `k` is greater than the number of data points `n` [conversation_history:1]

## Example Usage

