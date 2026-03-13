#!/usr/bin/env python3
"""DenseNet-121 architecture module based on Densely Connected CNNs (2018)."""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Build the DenseNet-121 architecture as described in Huang et al. (2018).

    Args:
        growth_rate: growth rate for the dense blocks
        compression: compression factor for the transition layers

    Returns:
        the keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    nb_filters = 2 * growth_rate

    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        nb_filters, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=init
    )(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
    x = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(x)

    model = K.models.Model(inputs=X, outputs=x)
    return model
