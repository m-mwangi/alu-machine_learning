#!/usr/bin/env python3
import numpy as np

class MultiNormal:
    def __init__(self, data):
        # Ensure data is a 2D numpy array
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        d, n = data.shape
        
        # Ensure there are multiple data points
        if n < 2:
            raise ValueError("data must contain multiple data points")
        
        # Calculate the mean
        self.mean = np.mean(data, axis=1, keepdims=True)
        
        # Calculate the covariance matrix without using np.cov
        centered_data = data - self.mean
        self.cov = (centered_data @ centered_data.T) / (n - 1)

