#!/usr/bin/env python3
"""Module for forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform forward propagation over a pooling layer."""
    # Unpack input shape
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Unpack kernel size and stride
    kh, kw = kernel_shape
    sh, sw = stride

    # Output size (no padding in pooling, always "valid")
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    # Output channels = input channels (pooling doesn't change channel count)
    A_out = np.zeros((m, h_out, w_out, c_prev))

    # Slide window across every output position
    for i in range(h_out):
        for j in range(w_out):
            # Convert output position to input position
            hs, ws = i * sh, j * sw

            # Grab the window: (m, kh, kw, c_prev)
            slice_ = A_prev[:, hs:hs+kh, ws:ws+kw, :]

            if mode == 'max':
                # Take max over the kh×kw window, keep batch and channel dims
                A_out[:, i, j, :] = np.max(slice_, axis=(1, 2))
            else:
                # Take mean over the kh×kw window
                A_out[:, i, j, :] = np.mean(slice_, axis=(1, 2))

    return A_out
