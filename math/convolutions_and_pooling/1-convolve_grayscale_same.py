#!/usr/bin/env python3
"""
Module for performing same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
            kh is the height of the kernel
            kw is the width of the kernel

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for same convolution
    ph = kh // 2
    pw = kw // 2

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Initialize output array (same dimensions as input)
    output = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            # Extract the region of interest
            region = padded_images[:, i:i+kh, j:j+kw]
            # Apply kernel and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
