#!/usr/bin/env python3
"""Importing modules"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes hue of an image
    image: 3D tensor
    delta: amount of hue to change
    """
    return tf.image.adjust_hue(image, delta)