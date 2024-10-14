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

    def pdf(self, x):
        """Calculates the PDF at a data point."""
        # Check if x is a numpy array
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        
        d, _ = self.mean.shape
        
        # Check if x has the correct shape
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")
        
        # Constants for the PDF calculation
        cov_inv = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det_cov)
        
        # Compute the PDF
        diff = x - self.mean
        exponent = -0.5 * (diff.T @ cov_inv @ diff)
        pdf_value = norm_factor * np.exp(exponent)
        
        return float(pdf_value)


