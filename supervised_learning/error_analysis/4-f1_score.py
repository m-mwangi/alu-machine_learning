#!/usr/bin/env python3
"""Importing modules"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates F1 score of a confusin matrix
    Returns: numpy array of shape(classes, )
    Containing F1 score
    """
    classes = confusion.shape[0]
    f1 = np.zeros(classes)
    precisions = precision(confusion)
    sensitivitys = sensitivity(confusion)
    for i in range(classes):
        f1[i] = 2 * (precisions[i] * sensitivitys[i]) / \
            (precisions[i] + sensitivitys[i])
    return f1
