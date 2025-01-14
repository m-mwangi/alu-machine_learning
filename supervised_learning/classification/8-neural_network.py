#!/usr/bin/env python3
"""Importing numpy"""


import numpy as np


class NeuralNetwork:
    """
    initializing the class
    """
    def __init__(self, nx, nodes):
        """
        Class constructor
        nx: no of input features
        nodes: no of nodes in hidden layer
        Attributes:
        W1: weights for hidden layer
        b1: bias of hidden layer
        A1: activated output for hidden layer
        W2: weights of output
        b2: bias for output
        A2: activated output
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
