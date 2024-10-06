#!/usr/bin/env python3
"""Class exponential"""


class Exponential:
    """
    Represents exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Constructor
        data is a list of the data to be used to estimate the distribution
        lambtha is the expected number of occurences in a given time frame
        Sets the instance attribute lambtha
        Saves lambtha as a float
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        x = time period
        """
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        x = time period
        """
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (-self.lambtha * x)
