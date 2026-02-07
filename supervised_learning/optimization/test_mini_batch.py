#!/usr/bin/env python3
import numpy as np
create_mini_batches = __import__('3-mini_batch').create_mini_batches

np.random.seed(0)
X = np.random.randn(100, 5)
Y = np.random.randn(100, 2)

batches = create_mini_batches(X, Y, 32)
print(f"Number of batches: {len(batches)}")
for i, (X_batch, Y_batch) in enumerate(batches):
    print(f"Batch {i}: X shape {X_batch.shape}, Y shape {Y_batch.shape}")
