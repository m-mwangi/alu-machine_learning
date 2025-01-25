#!/usr/bin/env python3
"""Importing tensorflow"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates training operation for network
    loss: loss of network
    alpha: learning rate
    returns: operation that trains using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)

    return train
