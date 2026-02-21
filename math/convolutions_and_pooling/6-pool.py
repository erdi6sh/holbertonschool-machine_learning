#!/usr/bin/env python3
"""
Module for performing pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel_shape: tuple of (kh, kw) containing the kernel shape for
                      the pooling
            kh is the height of the kernel
            kw is the width of the kernel
        stride: tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode: indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w, c))

    # Perform pooling
    for i in range(output_h):
        for j in range(output_w):
            # Calculate starting positions with stride
            h_start = i * sh
            w_start = j * sw
            # Extract the region of interest
            region = images[:, h_start:h_start+kh, w_start:w_start+kw, :]

            # Apply pooling operation
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
