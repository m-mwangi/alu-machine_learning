#!/usr/bin/env python3
"""Converts gensim word2vec to keras"""


def gensim_to_keras(model):
    """
    Converts gensim model to keras
    embedding layer
    Returns; trainable keras embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
