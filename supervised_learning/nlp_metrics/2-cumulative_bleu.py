#!/usr/bin/env python3
"""
Cumulative N-gram BLEU score calculation
"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list of list of str): reference translations
        sentence (list of str): model proposed sentence
        n (int): size of the largest n-gram to use

    Returns:
        float: cumulative n-gram BLEU score
    """
    precisions = []

    # Compute precision for each k-gram (1..n)
    for k in range(1, n + 1):
        # Candidate k-grams
        cand_ngrams = []
        for i in range(len(sentence) - k + 1):
            cand_ngrams.append(tuple(sentence[i:i + k]))

        cand_counts = {}
        for ng in cand_ngrams:
            cand_counts[ng] = cand_counts.get(ng, 0) + 1

        # Clip counts based on references
        clipped_counts = {}
        for ng in cand_counts:
            max_ref_count = 0
            for ref in references:
                ref_ngrams = []
                for i in range(len(ref) - k + 1):
                    ref_ngrams.append(tuple(ref[i:i + k]))
                ref_count = ref_ngrams.count(ng)
                if ref_count > max_ref_count:
                    max_ref_count = ref_count
            clipped_counts[ng] = min(cand_counts[ng], max_ref_count)

        # Precision for this k
        precision_k = sum(clipped_counts.values()) / max(1, len(cand_ngrams))
        precisions.append(precision_k)

    # Geometric mean of precisions (equal weights)
    if all(p > 0 for p in precisions):
        score = np.exp(sum(np.log(p) for p in precisions) / n)
    else:
        score = 0.0

    # Brevity penalty
    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - c), r))

    if c > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / c)

    # Final BLEU score
    bleu = bp * score
    return bleu
