#!/usr/bin/env python3
"""Module for back propagation over a pooling layer."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform back propagation over a pooling layer.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the output
            of the pooling layer.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
            containing the output of the previous layer.
        kernel_shape: tuple of (kh, kw) containing the size
            of the kernel for the pooling.
        stride: tuple of (sh, sw) containing the strides
            for the pooling.
        mode: string, either 'max' or 'avg', indicating
            whether to perform maximum or average pooling.

    Returns:
        numpy.ndarray: the partial derivatives with respect
            to the previous layer (dA_prev).
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            hs, ws = i * sh, j * sw
            if mode == 'max':
                slice_ = A_prev[:, hs:hs+kh, ws:ws+kw, :]
                mask = (slice_ == np.max(
                    slice_, axis=(1, 2), keepdims=True
                ))
                dA_prev[:, hs:hs+kh, ws:ws+kw, :] += (
                    mask * dA[:, i, j, :][:, None, None, :]
                )
            else:
                avg = dA[:, i, j, :][:, None, None, :] / (kh * kw)
                dA_prev[:, hs:hs+kh, ws:ws+kw, :] += (
                    np.ones((m, kh, kw, c)) * avg
                )

    return dA_prev
