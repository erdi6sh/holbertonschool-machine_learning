#!/usr/bin/env python3
"""Bag of Words embedding module."""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of vocabulary words to use. If None, all words are used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) with embeddings.
        features: list of features used for embeddings.
    """
    def tokenize(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", '', sentence)
        sentence = re.sub(r"[^a-z ]", '', sentence)
        return sentence.split()

    tokenized = [tokenize(s) for s in sentences]

    if vocab is None:
        all_words = set()
        for tokens in tokenized:
            all_words.update(tokens)
        features = sorted(all_words)
    else:
        features = list(vocab)

    word_index = {w: i for i, w in enumerate(features)}
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, tokens in enumerate(tokenized):
        for token in tokens:
            if token in word_index:
                embeddings[i][word_index[token]] += 1

    return embeddings, features
