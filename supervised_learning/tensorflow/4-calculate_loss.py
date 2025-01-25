#!/usr/bin/env python3
"""Importing tensorflow"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax crossentropy loss
    y- placeholder labels for imput
    y_pred: tensor with networks predictions
    Returns: Tensor containing loss
    """
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=y_pred))
    return loss
