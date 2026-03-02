#!/usr/bin/env python3
"""Module for back propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform back propagation over a convolutional layer."""
    # Unpack shapes of upstream gradient and inputs
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Recalculate same padding as in forward pass
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Re-pad input exactly as in forward pass (needed to compute dW)
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )

    # Initialize gradient arrays to zero
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)

    # db: bias was added to every (i,j) and every example → sum over all of them
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Loop over every output position
    for i in range(h_new):
        for j in range(w_new):
            hs, ws = i * sh, j * sw

            # Same receptive field slice used in forward pass
            slice_ = A_prev_pad[:, hs:hs+kh, ws:ws+kw, :]  # (m,kh,kw,c_prev)

            # Handle each filter k separately
            for k in range(c_new):
                # Upstream gradient at position (i,j) for filter k → (m,)
                dZ_ijk = dZ[:, i, j, k]

                # dW: slice_ was multiplied by W_k in forward
                # → gradient is slice_ * dZ_ijk, summed over batch
                dW[:, :, :, k] += np.sum(
                    slice_ * dZ_ijk[:, None, None, None], axis=0
                )

                # dA_prev: W_k was multiplied by input in forward
                # → scatter W_k * dZ_ijk back into input gradient
                # += because one input pixel can appear in multiple windows
                dA_prev_pad[:, hs:hs+kh, ws:ws+kw, :] += (
                    W[:, :, :, k] * dZ_ijk[:, None, None, None]
                )

    # Strip padding from gradient (padded zeros don't belong to original input)
    if ph > 0 and pw > 0:
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    elif ph > 0:
        dA_prev = dA_prev_pad[:, ph:-ph, :, :]
    elif pw > 0:
        dA_prev = dA_prev_pad[:, :, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
