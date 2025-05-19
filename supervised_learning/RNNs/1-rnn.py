#!/usr/bin/env python3
"""Importing numpy"""
import numpy as np


"""
Function that performs Forward propagation
"""


def rnn(rnn_cell, X, h_0):
    """
    rnn_cell-Instance of RNNCell
    X- Data used in shape(t, m, i)
    t- max no of time steps
    m- batch size
    i- dimensionality of data
    h_0-Initial hiddenstate shape(m, h)
    h- dimensionality of hiddenstate
    """
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t+1, m, h))
    H[0] = h_0

    for step in range(t):
        """
        Iterates over sequence to
        calculate forward prop
        """
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        if step == 0:
            """
            At the first step
            """
            Y = y
        else:
            Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)
    return (H, Y)
