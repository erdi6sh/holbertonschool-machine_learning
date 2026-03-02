#!/usr/bin/env python3
"""Module for back propagation over a pooling layer."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform back propagation over a pooling layer."""
    # Unpack shapes
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize input gradient (same shape as A_prev)
    dA_prev = np.zeros_like(A_prev)

    # Loop over every output position
    for i in range(h_new):
        for j in range(w_new):
            hs, ws = i * sh, j * sw

            # Same window used in forward pass
            slice_ = A_prev[:, hs:hs+kh, ws:ws+kw, :]  # (m,kh,kw,c)

            if mode == 'max':
                # Find the max value in each window (keepdims for broadcasting)
                max_vals = np.max(slice_, axis=(1, 2), keepdims=True)

                # Mask is True where the max occurred, False elsewhere
                mask = (slice_ == max_vals)

                # Only max positions receive the upstream gradient
                # reshape dA[:,i,j,:] from (m,c) to (m,1,1,c) for broadcasting
                dA_prev[:, hs:hs+kh, ws:ws+kw, :] += (
                    mask * dA[:, i, j, :][:, None, None, :]
                )
            else:
                # Avg pool: split gradient evenly across all kh*kw positions
                avg = dA[:, i, j, :][:, None, None, :] / (kh * kw)
                dA_prev[:, hs:hs+kh, ws:ws+kw, :] += (
                    np.ones((m, kh, kw, c)) * avg
                )

    return dA_prev
