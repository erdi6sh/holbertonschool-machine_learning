#!/usr/bin/env python3
"""Dense block module for DenseNet architecture."""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Build a dense block as described in Densely Connected CNNs (2018).

    Uses DenseNet-B bottleneck layers. Each layer applies BN -> ReLU -> 1x1
    Conv (producing 4*growth_rate feature maps) -> BN -> ReLU -> 3x3 Conv,
    then concatenates its output with all previous feature maps.

    Args:
        X: output from the previous layer
        nb_filters: number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs
    """
    init = K.initializers.HeNormal(seed=0)

    for _ in range(layers):
        x = K.layers.BatchNormalization(axis=3)(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            4 * growth_rate, (1, 1), padding='same', kernel_initializer=init
        )(x)

        x = K.layers.BatchNormalization(axis=3)(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same', kernel_initializer=init
        )(x)

        X = K.layers.concatenate([X, x])
        nb_filters += growth_rate

    return X, nb_filters
