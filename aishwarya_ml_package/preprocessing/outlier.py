"""Outlier detection and treatment using IQR and Z-Score."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class IQROutlierRemover(BaseEstimator):
    """
    Detect and cap outliers using the IQR method.
    Outlier if: x < Q1 - factor*IQR  or  x > Q3 + factor*IQR
    Strategy: 'clip' (Winsorize) or 'remove' rows.
    """
    def __init__(self, factor=1.5, strategy="clip"):
        self.factor   = factor
        self.strategy = strategy
        self.lower_   = None
        self.upper_   = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        Q1  = np.percentile(X, 25, axis=0)
        Q3  = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        self._check_fitted("lower_")
        X = np.array(X, dtype=np.float64).copy()
        if self.strategy == "clip":
            return np.clip(X, self.lower_, self.upper_)
        elif self.strategy == "remove":
            mask = np.all((X >= self.lower_) & (X <= self.upper_), axis=1)
            return X[mask]
        raise ValueError("strategy must be 'clip' or 'remove'")

    def fit_transform(self, X, y=None):
        self.fit(X)
        if self.strategy == "remove" and y is not None:
            X_arr = np.array(X, dtype=np.float64)
            mask  = np.all(
                (X_arr >= self.lower_) & (X_arr <= self.upper_), axis=1)
            return X_arr[mask], np.array(y)[mask]
        return self.transform(X)

    def outlier_report(self, X):
        """Print how many outliers exist per feature."""
        X = np.array(X, dtype=np.float64)
        self._check_fitted("lower_")
        print(f"{'Feature':<8} {'Lower':>10} {'Upper':>10} {'Outliers':>10}")
        print("-" * 42)
        for j in range(X.shape[1]):
            n_out = ((X[:, j] < self.lower_[j]) |
                     (X[:, j] > self.upper_[j])).sum()
            print(f"{'X'+str(j):<8} {self.lower_[j]:>10.3f} "
                  f"{self.upper_[j]:>10.3f} {n_out:>10}")


class ZScoreOutlierRemover(BaseEstimator):
    """
    Detect outliers using Z-Score.
    Outlier if |z| > threshold  (default = 3).
    """
    def __init__(self, threshold=3.0, strategy="clip"):
        self.threshold = threshold
        self.strategy  = strategy
        self.mean_     = None
        self.std_      = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        self._check_fitted("mean_")
        X = np.array(X, dtype=np.float64).copy()
        z = np.abs((X - self.mean_) / self.std_)
        if self.strategy == "clip":
            lo = self.mean_ - self.threshold * self.std_
            hi = self.mean_ + self.threshold * self.std_
            return np.clip(X, lo, hi)
        elif self.strategy == "remove":
            mask = np.all(z <= self.threshold, axis=1)
            return X[mask]
        raise ValueError("strategy must be 'clip' or 'remove'")

    def fit_transform(self, X, y=None):
        self.fit(X)
        if self.strategy == "remove" and y is not None:
            X_arr = np.array(X, dtype=np.float64)
            z     = np.abs((X_arr - self.mean_) / self.std_)
            mask  = np.all(z <= self.threshold, axis=1)
            return X_arr[mask], np.array(y)[mask]
        return self.transform(X)
