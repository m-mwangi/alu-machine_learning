#!/usr/bin/env python3
"""Importing necessary libraries"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and preps dataset for machine
    translation
    """
    def __init__(self, batch_size, max_len):
        """
        Creates the instance attributes:
            data_train, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset train split, loaded as_supervided
            data_valid, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt is the Portuguese tokenizer created from the training set
            tokenizer_en is the English tokenizer created from the training set
        """
        # Load the train and validation splits of the dataset
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation',
            as_supervised=True
        )

        # Create the tokenizers using the tokenize_dataset method
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Update data_train and data_valid by tokenizing the examples
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Filter, cache, shuffle, batch, and prefetch data_train
        def filter_max_len(x, y):
            return tf.logical_and(tf.size(x) <= max_len, tf.size(y) <= max_len)

        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(
            tf.data.experimental.cardinality(self.data_train).numpy())
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)


        # Filter and batch data_valid
        self.data_valid = self.data_valid.filter(filter_max_len)
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.

        Args:
            data: examples are formatted as a tuple (pt, en)
            pt: the tf.Tensor containing the Portuguese sentence
            en: the tf.Tensor containing the corresponding English sentence

        Returns:
            tokenizer_pt: the Portuguese tokenizer
            tokenizer_en: the English tokenizer
        """
        # Build the Portuguese tokenizer
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15
        )

        # Build the English tokenizer
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes translation into tokens
        pr: Tensor containing Portuguese sentence
        en: Tensor containing English sentence
        Tokenized sentence include start and end tokens
        start token: indexed as vocab size
        end token: indexed as vocab size + 1
        Returns:
            pt_tokens: list of Portuguese tokens
            en_tokens: list of English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')
        ) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy().decode('utf-8')
        ) + [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as tf wrapper to encode instance
        method.
        Set shape of pt and en return tensor
        """
        pt_result, en_result = tf.py_function(self.encode, [pt, en],
         [tf.int64, tf.int64])
        pt_result.set_shape([None])
        en_result.set_shape([None])
        return pt_result, en_result
