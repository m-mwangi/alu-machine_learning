#!/usr/bin/env python3

"""This module contains a function that
returns tensor output of the layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev - tensor output of previous layer
    n - no. of nodes in layer to create
    activation - activation function
    layer - name of layers
    """
    # He et al. initializer for the layer weights
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    # Create the layer
    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name=None)
    return layer
