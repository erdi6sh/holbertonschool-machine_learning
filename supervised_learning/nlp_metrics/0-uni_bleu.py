#!/usr/bin/env python3
"""
Unigram BLEU score calculation
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list of list of str): reference translations
        sentence (list of str): model proposed sentence

    Returns:
        float: unigram BLEU score
    """
    # Count words in candidate sentence
    word_counts = {}
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Clip counts based on references
    clipped_counts = {}
    for word in word_counts:
        max_ref_count = 0
        for ref in references:
            ref_count = ref.count(word)
            if ref_count > max_ref_count:
                max_ref_count = ref_count
        clipped_counts[word] = min(word_counts[word], max_ref_count)

    # Precision: sum of clipped counts / total candidate words
    precision = sum(clipped_counts.values()) / len(sentence)

    # Brevity penalty
    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - c), r))

    if c > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / c)

    # BLEU score
    bleu = bp * precision
    return bleu
