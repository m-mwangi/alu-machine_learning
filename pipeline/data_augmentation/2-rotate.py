#!/usr/bin/env python3
"""Importing modules"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates image by 90 degrees
    counterclockwise
    """
    return tf.image.rot90(image)