#!/usr/bin/env python3
"""Learning rate decay using inverse time decay"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow
    using inverse time decay

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate at which
                    alpha will decay
        decay_step: the number of passes of gradient descent
                    before alpha decays

    Returns:
        The learning rate decay operation
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
