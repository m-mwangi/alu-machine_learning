#!/usr/bin/env python3
"""Importing numpy"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.
    """
    t, m, i = X.shape  # time steps, batch size, input dimensions
    h = h_0.shape[1]  # hidden state size

    # Initialize the hidden states and outputs
    H_forward = np.zeros((t + 1, m, h))
    H_backward = np.zeros((t + 1, m, h))
    H_forward[0] = h_0  # Forward direction initial state
    H_backward[-1] = h_t  # Backward direction initial state

    # Forward pass
    for step in range(t):
        H_forward[step + 1] = bi_cell.forward(H_forward[step], X[step])

    # Backward pass
    for step in range(t - 1, -1, -1):
        H_backward[step] = bi_cell.backward(H_backward[step + 1], X[step])

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_forward[1:], H_backward[:-1]), axis=-1)

    # Compute outputs
    Y = bi_cell.output(H)
    return H, Y
