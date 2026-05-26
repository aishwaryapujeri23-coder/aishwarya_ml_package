"""Scalers: StandardScaler and MinMaxScaler — built from scratch."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class StandardScaler(BaseEstimator):
    """
    Standardize features: z = (x - mean) / std
    Result: mean=0, std=1 for every feature.

    Usage
    -----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    """
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        X = self._validate_X_y(X)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1   # avoid division by zero
        return self

    def transform(self, X):
        self._check_fitted("mean_")
        X = self._validate_X_y(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        self._check_fitted("mean_")
        return X * self.std_ + self.mean_


class MinMaxScaler(BaseEstimator):
    """
    Scale features to [0, 1] range: x_norm = (x - min) / (max - min)

    Usage
    -----
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_  = None
        self.max_  = None

    def fit(self, X):
        X = self._validate_X_y(X)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        self._check_fitted("min_")
        X   = self._validate_X_y(X)
        rng = self.max_ - self.min_
        rng[rng == 0] = 1
        lo, hi = self.feature_range
        return lo + (X - self.min_) / rng * (hi - lo)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        self._check_fitted("min_")
        lo, hi = self.feature_range
        rng = self.max_ - self.min_
        return self.min_ + (X - lo) / (hi - lo) * rng
