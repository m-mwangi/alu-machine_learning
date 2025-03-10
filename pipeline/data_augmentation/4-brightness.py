#!/usr/bin/env python3
"""importing modules"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly change brightness of an image
    image: 3D tensor
    max_delta: max amount of image to brighten
    returns altered image
    """
    return tf.image.random_brightness(image, max_delta)