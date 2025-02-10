#!/usr/bin/env python3
"""Importing numpy"""
import numpy as np


def precision(confusion):
    """
    Calculates precision for each class
    Returns: np array containing precision of each class
    """
    classes = confusion.shape[0]
    precision = np.zeros(classes)
    for i in range(classes):
        precision[i] = confusion[i, i] / np.sum(confusion[:, i])
    return precision
