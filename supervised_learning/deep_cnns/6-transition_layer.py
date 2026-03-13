#!/usr/bin/env python3
"""Transition layer module for DenseNet architecture."""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Build a transition layer as described in Densely Connected CNNs (2018).

    Implements DenseNet-C compression: reduces feature maps by compression
    factor using BN -> ReLU -> 1x1 Conv -> 2x2 Average Pooling.

    Args:
        X: output from the previous layer
        nb_filters: number of filters in X
        compression: compression factor for the transition layer

    Returns:
        output of the transition layer and the number of filters within output
    """
    init = K.initializers.HeNormal(seed=0)
    nb_filters = int(nb_filters * compression)

    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        nb_filters, (1, 1), padding='same', kernel_initializer=init
    )(x)
    x = K.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='valid')(x)

    return x, nb_filters
