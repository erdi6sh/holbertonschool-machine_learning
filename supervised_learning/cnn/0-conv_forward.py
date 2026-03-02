#!/usr/bin/env python3
"""Module for forward propagation over a convolutional layer."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Perform forward propagation over a convolutional layer."""
    # Unpack input shape: m=examples, h/w=spatial dims, c_prev=input channels
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Unpack filter shape: kh/kw=filter size, c_new=number of filters
    kh, kw, _, c_new = W.shape

    # Unpack stride into height-stride and width-stride
    sh, sw = stride

    # Calculate padding needed on each side
    if padding == "same":
        # Formula keeps output size ≈ input size / stride
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        # "valid" = no padding, output will shrink
        ph, pw = 0, 0

    # Output size formula: (input + 2*pad - kernel) / stride + 1
    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    # Add zero-padding to height and width only (not batch or channels)
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )

    # Pre-allocate output array (pre-activation values)
    Z = np.zeros((m, h_out, w_out, c_new))

    # Slide filter across every output position (i, j)
    for i in range(h_out):
        for j in range(w_out):
            # Convert output position to input position using stride
            hs, ws = i * sh, j * sw

            # Grab the input patch this output cell "sees" → (m, kh, kw, c_prev)
            slice_ = A_prev_pad[:, hs:hs+kh, ws:ws+kw, :]

            # Dot product over (kh, kw, c_prev) axes → (m, c_new), then add bias
            Z[:, i, j, :] = np.tensordot(
                slice_, W, axes=([1, 2, 3], [0, 1, 2])
            ) + b[0, 0, 0, :]

    # Apply activation function (e.g. ReLU) element-wise and return
    return activation(Z)
