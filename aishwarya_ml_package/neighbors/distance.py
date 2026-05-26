"""Distance functions used by KNN."""
import numpy as np


def manhattan(a, b):
    """City Block distance: sum(|a_i - b_i|). Shape: (n_train,)"""
    return np.sum(np.abs(b - a), axis=1)


def euclidean(a, b):
    """Euclidean distance: sqrt(sum((a_i-b_i)^2))"""
    return np.sqrt(np.sum((b - a) ** 2, axis=1))


def cosine(a, b):
    """Cosine distance: 1 - cosine_similarity."""
    dot  = b @ a
    norm = np.linalg.norm(b, axis=1) * np.linalg.norm(a)
    norm[norm == 0] = 1
    return 1 - dot / norm
