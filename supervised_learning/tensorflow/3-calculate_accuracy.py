#!/usr/bin/env python3
"""Importing tensorflow"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates accuracy of a prediction
    y: labels of input data
    y_pred: tensor with network's predictions
    Returns: Tensor contain decimal accuracy
    hint: accuracy = correct_predictions / all_predictions
    """

    # Return index of largest value(onehotencode)
    predicted_labels = tf.argmax(y_pred, 1)
    true_labels = tf.argmax(y, 1)

    # Compare the true and predicted labels
    predictions_check = tf.equal(predicted_labels, true_labels)

    # Convert Boolean to float then compute mean
    accuracy = tf.reduce_mean(tf.cast(predictions_check, tf.float32))
    return accuracy
