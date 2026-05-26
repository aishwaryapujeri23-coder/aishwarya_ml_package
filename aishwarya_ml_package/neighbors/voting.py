"""Voting strategies for KNN prediction."""
import numpy as np


def majority_vote(labels):
    """Return most common label among neighbors."""
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]


def weighted_vote(labels, distances):
    """
    Weighted vote: closer neighbors count more.
    Weight = 1 / (distance + eps)
    """
    eps     = 1e-8
    weights = 1.0 / (distances + eps)
    classes = np.unique(labels)
    scores  = {c: 0.0 for c in classes}
    for label, w in zip(labels, weights):
        scores[label] += w
    return max(scores, key=scores.get)
