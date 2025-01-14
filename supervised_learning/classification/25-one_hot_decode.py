#!/usr/bin/env python3
"""Importing necessary module"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts one hot matric into vector
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    numeric_labels = np.argmax(one_hot, axis=0)
    return numeric_labels
