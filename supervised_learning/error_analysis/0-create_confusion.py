#!/usr/bin/env python3
"""Importing numpy"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Function hat creates a confusion matrix
    labels: np.array with correct labels
    logits: array containing predicted labels
    returns: Confusion with row(correct labels)
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for i in range(m):
        confusion[np.argmax(labels[i]), np.argmax(logits[i])] += 1
    return confusion
