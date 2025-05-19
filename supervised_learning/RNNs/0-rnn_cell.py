#!/usr/bin/env python3
"""Importinf numpy"""
import numpy as np


class RNNCell:
    """
    Represents a cell of simple RNN
    """
    def __init__(self, i, h, o):
        """
        i = dimensionality of data
        h - dimensionality of hdden state
        o - dimensionality of output
        Attributes:
        Wh & bh- concat hidden state and input
        Wy and by- output
        Biases- initialize to 0
        Weights- Random normal distribution
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Performs the softmax function
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def forward(self, h_prev, x_t):
        """
        Peforms forward propagation
        """
        concatenation = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
