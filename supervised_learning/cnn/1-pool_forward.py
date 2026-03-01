"""Module for forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform forward propagation over a pooling layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        kernel_shape: tuple of (kh, kw) containing the size
            of the kernel for the pooling.
        stride: tuple of (sh, sw) containing the strides
            for the pooling.
        mode: string, either 'max' or 'avg', indicating
            whether to perform maximum or average pooling.

    Returns:
        numpy.ndarray: the output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    A_out = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            hs, ws = i * sh, j * sw
            slice_ = A_prev[:, hs:hs+kh, ws:ws+kw, :]
            if mode == 'max':
                A_out[:, i, j, :] = np.max(slice_, axis=(1, 2))
            else:
                A_out[:, i, j, :] = np.mean(slice_, axis=(1, 2))

    return A_out
