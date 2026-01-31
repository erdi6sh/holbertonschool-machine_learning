#!/usr/bin/env python3
"""Module for making predictions with a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network

    Args:
        network: the network model to make the prediction with
         the input data to make the prediction with
        verbose: boolean that determines if output should be printed

    Returns:
        the prediction for the data
    """
    prediction = network.predict(x=data, verbose=verbose)
    return prediction
