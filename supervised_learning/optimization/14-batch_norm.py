#!/usr/bin/env python3
"""Module for creating batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function for the output of the layer

    Returns:
        A tensor of the activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )
    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    mean, variance = tf.nn.moments(Z, axes=[0])
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon
    )

    return activation(Z_norm)
