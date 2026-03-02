#!/usr/bin/env python3
"""Module for building a modified LeNet-5 architecture using Keras."""
from tensorflow import keras as K


def lenet5(X):
    """Build and compile a modified LeNet-5 model."""
    # He Normal init: scales weights for ReLU networks, seed=0 for reproducibility
    init = K.initializers.HeNormal(seed=0)

    # Conv layer 1: 6 filters, 5×5 kernel, same padding → output stays 28×28
    x = K.layers.Conv2D(
        6, (5, 5), padding='same', activation='relu', kernel_initializer=init
    )(X)

    # MaxPool 1: 2×2 window, stride 2 → halves size: 28×28 → 14×14
    x = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Conv layer 2: 16 filters, 5×5, valid padding → size shrinks: 14×14 → 10×10
    x = K.layers.Conv2D(
        16, (5, 5), padding='valid', activation='relu', kernel_initializer=init
    )(x)

    # MaxPool 2: halves size again: 10×10 → 5×5, now shape is (m, 5, 5, 16)
    x = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Flatten: convert (m, 5, 5, 16) → (m, 400) to feed into dense layers
    x = K.layers.Flatten()(x)

    # Fully connected layer with 120 nodes + ReLU
    x = K.layers.Dense(120, activation='relu', kernel_initializer=init)(x)

    # Fully connected layer with 84 nodes + ReLU
    x = K.layers.Dense(84, activation='relu', kernel_initializer=init)(x)

    # Output layer: 10 nodes (one per digit), softmax gives class probabilities
    y = K.layers.Dense(10, activation='softmax', kernel_initializer=init)(x)

    # Wrap into a Model object connecting input X to output y
    model = K.Model(inputs=X, outputs=y)

    # Compile: adam optimizer, cross-entropy loss for one-hot labels, track accuracy
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
