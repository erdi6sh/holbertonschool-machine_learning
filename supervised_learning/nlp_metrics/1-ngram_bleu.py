#!/usr/bin/env python3
"""
N-gram BLEU score calculation
"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list of list of str): reference translations
        sentence (list of str): model proposed sentence
        n (int): size of the n-gram to use

    Returns:
        float: n-gram BLEU score
    """
    # Build candidate n-grams
    cand_ngrams = []
    for i in range(len(sentence) - n + 1):
        cand_ngrams.append(tuple(sentence[i:i + n]))

    # Count candidate n-grams
    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    # Clip counts based on references
    clipped_counts = {}
    for ng in cand_counts:
        max_ref_count = 0
        for ref in references:
            ref_ngrams = []
            for i in range(len(ref) - n + 1):
                ref_ngrams.append(tuple(ref[i:i + n]))
            ref_count = ref_ngrams.count(ng)
            if ref_count > max_ref_count:
                max_ref_count = ref_count
        clipped_counts[ng] = min(cand_counts[ng], max_ref_count)

    # Precision
    precision = sum(clipped_counts.values()) / max(1, len(cand_ngrams))

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
