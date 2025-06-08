#!/usr/bin/env python3
"""Importing necessary libraries"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and preps dataset for machine
    translation
    """
    def __init__(self):
        """
        Creates instance attributes:
        data_train: contain train split loaded
        as_supervised
        data_valid: validate split loaded
        as_supervised
        tokenizer_pt: Portuguese tokenizer
        tokenizer_en: English tokenizer
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train', as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation',
            as_supervised=True
        )

        # Create the tokenizers using the tokenize_dataset method
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

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
