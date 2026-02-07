#!/usr/bin/env python3
"""Create and save a simple MNIST model"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset
lib = np.load('MNIST.npz')
X_train = lib['X_train']
Y_train = lib['Y_train']

# Normalize the data
X_train = X_train.astype('float32') / 255.0

# Flatten the images
X_train_flat = X_train.reshape((X_train.shape[0], -1))

# Convert labels to one-hot encoding
Y_train_oh = keras.utils.to_categorical(Y_train, 10)

# Build a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model (this may take a few minutes)...")
# Train the model briefly
model.fit(X_train_flat, Y_train_oh, epochs=5, batch_size=128, verbose=1)

# Save the model
model.save('model.h5')
print("\nModel saved successfully to model.h5")
print(f"Model summary:")
model.summary()
