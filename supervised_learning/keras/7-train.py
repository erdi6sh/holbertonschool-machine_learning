#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
         numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) with labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_ data to validate the model with, if not None
        early_stopping: boolean indicating if early stopping should be used
        patience: patience used for early stopping
        learning_rate_decay: boolean indicating if learning rate decay
        alpha: initial learning rate
        decay_rate: decay rate
        verbose: boolean that determines if output should be printed
        shuffle: boolean determining if batches should be shuffled

    Returns:
        the History object generated after training the model
    """
    callbacks = []

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            """Learning rate schedule function using inverse time decay"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(
            schedule,
            verbose=1
        )
        callbacks.append(lr_decay)

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
