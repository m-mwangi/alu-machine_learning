#!/usr/bin/env python3
"""Importing numpy"""


import numpy as np


class Neuron:
    """Initializing the class"""
    def __init__(self, nx):
        """
        nx: number of input features
        Private instance attributes:
        __W: weights vector for neuron
        __b: bias of neuron
        __A: activated output
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter function"""
        return self.__W

    @property
    def b(self):
        """Getter function b"""
        return self.__b

    @property
    def A(self):
        """Getter function A"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculate forward prop
        X: np.ndarray with shape(nx,m)
        nx: no of input features
        m: no of examples
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
