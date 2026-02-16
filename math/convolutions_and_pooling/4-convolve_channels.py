#!/usr/bin/env python3
"""Module for convolution with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.
    
    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)
    
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    
    # Calculate padding
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + (kh % 2 == 0)
        pw = int(((w - 1) * sw + kw - w) / 2) + (kw % 2 == 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    
    # Pad images
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 
                          mode='constant', constant_values=0)
    
    # Calculate output dimensions
    output_h = int((h + 2 * ph - kh) / sh) + 1
    output_w = int((w + 2 * pw - kw) / sw) + 1
    
    # Initialize output
    output = np.zeros((m, output_h, output_w))
    
    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # Extract the region
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            
            # Get the image region
            region = images_padded[:, h_start:h_end, w_start:w_end, :]
            
            # Apply kernel across all channels and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))
    
    return output
