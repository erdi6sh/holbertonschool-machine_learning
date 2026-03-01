#!/usr/bin/env python3
"""Module for back propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform back propagation over a convolutional layer.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated
            output of the convolutional layer.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution.
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing
            the biases applied to the convolution.
        padding: string, either 'same' or 'valid',
            indicating the type of padding used.
        stride: tuple of (sh, sw) containing the strides
            for the convolution.

    Returns:
        tuple: partial derivatives with respect to the previous layer
            (dA_prev), the kernels (dW), and the biases (db).
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            hs, ws = i * sh, j * sw
            slice_ = A_prev_pad[:, hs:hs+kh, ws:ws+kw, :]
            for k in range(c_new):
                dZ_ijk = dZ[:, i, j, k]
                dW[:, :, :, k] += np.sum(
                    slice_ * dZ_ijk[:, None, None, None], axis=0
                )
                dA_prev_pad[:, hs:hs+kh, ws:ws+kw, :] += (
                    W[:, :, :, k] * dZ_ijk[:, None, None, None]
                )

    if ph > 0 and pw > 0:
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    elif ph > 0:
        dA_prev = dA_prev_pad[:, ph:-ph, :, :]
    elif pw > 0:
        dA_prev = dA_prev_pad[:, :, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
