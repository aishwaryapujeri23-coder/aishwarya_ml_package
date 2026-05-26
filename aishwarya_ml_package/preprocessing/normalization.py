"""Row-wise normalizers (L1, L2)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class L2Normalizer(BaseEstimator):
    """Scale each sample so its L2 norm = 1. Used in KNN / cosine similarity."""
    def fit(self, X):
        return self   # stateless

    def transform(self, X):
        X    = np.array(X, dtype=np.float64)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return X / norm

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class L1Normalizer(BaseEstimator):
    """Scale each sample so its L1 norm = 1."""
    def fit(self, X):
        return self

    def transform(self, X):
        X    = np.array(X, dtype=np.float64)
        norm = np.abs(X).sum(axis=1, keepdims=True)
        norm[norm == 0] = 1
        return X / norm

    def fit_transform(self, X):
        return self.fit(X).transform(X)
