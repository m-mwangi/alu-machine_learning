#!/usr/bin/env python3
"""importing modules"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    Randomly shears an image
    image: 3D tensor containing image to shear
    intensity: intensity the shearing should happen
    """
    image_np = image.numpy()
    sheared_image_np = tf.keras.preprocessing.image.random_shear(
        image_np, intensity, row_axis=0, col_axis=1, channel_axis=2
    )
    return tf.convert_to_tensor(sheared_image_np)