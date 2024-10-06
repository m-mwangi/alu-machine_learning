#!/usr/bin/env python3
"""Creating a class Poisson"""


class Poisson:
    """
    represents a poisson distribution
    data is a list of the data to be used to estimate the distribution
    lambtha is the expected number of occurences in a given time frame
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Sets the instance attribute lambtha
        Saves lambtha as a float
        If data is not given, (i.e. None
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(1, k + 1):
            factorial = factorial * i
        result = self.lambtha ** k * 2.7182818285 ** (-self.lambtha)
        return result / factorial

    def cdf(self, k):
        """
        Calculate the value of the CDF for a given number of “successes”
        k = "successes"
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        result = 0
        for i in range(k + 1):
            result = result + self.pmf(i)
        return result
