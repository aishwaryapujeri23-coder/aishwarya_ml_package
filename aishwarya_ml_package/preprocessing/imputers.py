"""Imputers for handling missing values (NaN)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class MeanImputer(BaseEstimator):
    """Replace NaN with column mean."""
    def __init__(self):
        self.fill_values_ = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.fill_values_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        self._check_fitted("fill_values_")
        X = np.array(X, dtype=np.float64).copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.fill_values_[j]
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MedianImputer(BaseEstimator):
    """Replace NaN with column median. Better for skewed data / outliers."""
    def __init__(self):
        self.fill_values_ = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.fill_values_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        self._check_fitted("fill_values_")
        X = np.array(X, dtype=np.float64).copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.fill_values_[j]
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ModeImputer(BaseEstimator):
    """Replace NaN with column mode. Best for categorical / integer data."""
    def __init__(self):
        self.fill_values_ = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        modes = []
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            vals, counts = np.unique(col, return_counts=True)
            modes.append(vals[np.argmax(counts)])
        self.fill_values_ = np.array(modes)
        return self

    def transform(self, X):
        self._check_fitted("fill_values_")
        X = np.array(X, dtype=np.float64).copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.fill_values_[j]
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
