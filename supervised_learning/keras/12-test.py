#!/usr/bin/env python3
"""Module for testing a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Args:
        network: the network model to test
         the input data to test the model with
        labels: the correct one-hot labels of data
        verbose: boolean that determines if output should be printed

    Returns:
        the loss and accuracy of the model with the testing data
    """
    results = network.evaluate(x=data, y=labels, verbose=verbose)
    return results
