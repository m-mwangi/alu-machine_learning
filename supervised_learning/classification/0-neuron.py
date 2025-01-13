#!/usr/bin/env python3
"""Importing numpy"""


import numpy as np


class Neuron():
    """
    initializing the class
    """
    def __init__(self, nx):
        """
        nx: no of input features
        W: weights vector
        b: bias of neuron
        A: activated output of neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
