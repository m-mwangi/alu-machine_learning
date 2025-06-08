#!/usr/bin/env python3
"""Importing necessary libraries"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    input: Tensor of shape (batch_size, seq_len)
    target: Tensor of shape (batch_size, seq_len)
    Returns:
        encoder_mask: tf.Tensor of shape (batch_size, 1, 1, seq_len)
        combined_mask: tf.Tensor of shape (batch_size, 1, seq_len, seq_len)
        used in 1st attention block in decoder
        decoder_mask: tf.Tensor of shape (batch_size, 1, seq_len, seq_len)
        used in 2nd attention block in decoder
    """
    def create_padding_mask(seq):
        """Creates a padding mask for a sequence."""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(size):
        """Creates a look-ahead mask for a sequence."""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)
    combined_mask = tf.maximum(create_padding_mask(target), create_look_ahead_mask(tf.shape(target)[1]))

    return encoder_mask, combined_mask, decoder_mask
