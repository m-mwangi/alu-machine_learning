#!/usr/bin/env python3
"""Importing modules"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs random crop of an image
    """
    return tf.image.random_crop(image,
                                size=(size[0], size[1], image.shape[-1]))