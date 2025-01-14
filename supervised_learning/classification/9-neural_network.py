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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter function"""
        return self.__W1

    @property
    def b1(self):
        """Getter function b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter function A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter function W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter function for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter function A2"""
        return self.__A2
