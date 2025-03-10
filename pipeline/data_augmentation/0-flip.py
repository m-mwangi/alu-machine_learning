#!/usr/bin/env python3
"""importing modules"""
import tensorflow as tf


def flip_image(image):
    """
    Flips and image horizontally
    """
    return tf.image.flip_left_right(image)