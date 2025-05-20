#!/usr/bin/env python3
"""Creating bOW"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates BOW embedding matrix
    sentences; list of sen to analyze
    vocab; list of vocab words for analysis
    Returns; embeddings, features
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X_train_counts = vectorizer.fit_transform(sentences)
    embeddings = X_train_counts.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
