"""Encoders for categorical variables."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class LabelEncoder(BaseEstimator):
    """
    Convert string categories → integer labels.
    e.g. ['cat','dog','cat'] → [0, 1, 0]
    """
    def __init__(self):
        self.classes_   = None
        self._label_map = None

    def fit(self, y):
        self.classes_   = np.unique(y)
        self._label_map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        self._check_fitted("classes_")
        return np.array([self._label_map[v] for v in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        self._check_fitted("classes_")
        return np.array([self.classes_[int(i)] for i in y])


class OneHotEncoder(BaseEstimator):
    """
    Convert integer labels → binary matrix (one column per category).
    e.g. [0,1,2] → [[1,0,0],[0,1,0],[0,0,1]]
    """
    def __init__(self, sparse=False):
        self.sparse   = sparse
        self.n_cats_  = None
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.n_cats_  = len(self.classes_)
        return self

    def transform(self, y):
        self._check_fitted("n_cats_")
        y   = np.array(y, dtype=int).ravel()
        ohe = np.zeros((len(y), self.n_cats_), dtype=np.float64)
        ohe[np.arange(len(y)), y] = 1.0
        return ohe

    def fit_transform(self, y):
        return self.fit(y).transform(y)
