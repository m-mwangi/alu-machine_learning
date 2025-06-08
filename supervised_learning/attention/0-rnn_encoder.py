#!/usr/bin/env python3
"""Importing libraries"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        vocab-int, size of input vocab
        embedding-embedding vector
        units-hidden units in RNN cell
        batch-batch size
        """
        if type(vocab) is not int:
            raise TypeError(
                "vocab must be int representing the size of input vocabulary"
            )
        if type(embedding) is not int:
            raise TypeError(
                "embedding must be int representing dimensionality of vector"
            )
        if type(units) is not int:
            raise TypeError(
                "units must be int representing the number of hidden units"
            )
        if type(batch) is not int:
            raise TypeError("batch must be int representing the batch size")
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def initialize_hidden_state(self):
        """
        Initializes hi to tensor(0s)
        """
        hidden_states = tf.zeros(shape=(self.batch, self.units))
        return hidden_states

    def call(self, x, initial):
        """
        x:f shape(batch, inp_seq_len)
        contain input to encoded layer
        initial: initial hidden state
        Returns:
        outputs(batch, in_seq_len, units)
        hidden(batch, units)
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
