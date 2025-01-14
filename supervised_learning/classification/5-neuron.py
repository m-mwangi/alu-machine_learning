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

    def cost(self, Y, A):
        """
        Calculates cost of model using
        logistic regression
        Y: labels for input data with shape(1, m)
        A: Activated output for neuron
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Returns neuron prediction and cost
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        calculates one pass of gradient descent
        alpha: learning rate
        """
        m = Y.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(X, dz.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db
