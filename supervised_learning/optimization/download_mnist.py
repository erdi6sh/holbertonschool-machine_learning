#!/usr/bin/env python3
"""Download and save MNIST dataset"""
import numpy as np
import tensorflow as tf

# Load MNIST dataset from Keras
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# Save to MNIST.npz file
np.savez_compressed('MNIST.npz', 
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test,
                    Y_test=Y_test)

print("MNIST dataset saved successfully to MNIST.npz")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")
