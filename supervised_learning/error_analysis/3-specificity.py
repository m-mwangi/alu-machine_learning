#!/usr/bin/env python3
"""Importing numpy"""
import numpy as np


def specificity(confusion):
    """
    Calculates specificity for each class
    Returns: np array containing specificity
    """
    classes = confusion.shape[0]  # will be the same as the number of rows
    specificity = np.zeros(classes)  # create an array of zeros
    for i in range(classes):
        true_negatives = np.sum(confusion) - np.sum(
            confusion[i]) - np.sum(confusion[:, i]) + confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        false_negatives = np.sum(confusion[i]) - confusion[i, i]
        true_positives = confusion[i, i]
        specificity[i] = true_negatives / (true_negatives + false_positives)
    return specificity
