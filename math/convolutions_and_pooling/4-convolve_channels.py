#!/usr/bin/env python3
"""Module for convolution with channels"""
import numpy as np

#Takes an image and slides a filter (kernel) across it to detect patterns like edges, blur, or sharpen.
def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
        kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape # m images, h×w pixels, c colors
    kh, kw, kc = kernel.shape # kernel is kh×kw size
    sh, sw = stride

    if padding == 'same': # Add border so output = input size
        ph = int(((h - 1) * sh + kh - h) / 2) + (kh % 2 == 0)  # pixels to add top/bottom
        pw = int(((w - 1) * sw + kw - w) / 2) + (kw % 2 == 0)
    elif padding == 'valid':
        ph, pw = 0, 0 # No border, output shrinks
    else:
        ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), # Add zeros around edges
                           mode='constant', constant_values=0)

    output_h = int((h + 2 * ph - kh) / sh) + 1 # How many positions kernel fits
    output_w = int((w + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h): # Move down
        for j in range(output_w): # Move right
            h_start = i * sh # Starting row
            h_end = h_start + kh # Ending row
            w_start = j * sw # Starting column
            w_end = w_start + kw # Ending column

            region = images_padded[:, h_start:h_end, w_start:w_end, :] # Grab pixels under kernel

            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3)) # Multiply & sum

    return output
