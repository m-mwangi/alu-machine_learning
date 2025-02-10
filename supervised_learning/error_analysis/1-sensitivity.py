#!/usr/bin/env python3
"""Importing numpy"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates sensitivity for each class in confusion
    confusion: (classes, classes)
    classes: no of classes
    Returns: np.array shape(classes,) containing
    sensitivity
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)
    for i in range(classes):
        sensitivity[i] = confusion[i, i] / np.sum(confusion[i])
    return sensitivity
