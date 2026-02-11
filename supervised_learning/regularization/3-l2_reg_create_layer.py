#!/usr/bin/env python3
"""Module for creating a layer with L2 regularization."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the layer
        lambtha: L2 regularization parameter

    Returns:
        Output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode=('fan_avg')
    )

    regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
