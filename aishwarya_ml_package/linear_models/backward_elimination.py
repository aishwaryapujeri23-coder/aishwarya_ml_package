"""Backward Elimination based on p-value threshold."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class BackwardElimination(BaseEstimator):
    """
    Remove features with p-value > threshold one at a time.
    Uses OLS t-test p-values to decide which to remove.
    """
    def __init__(self, threshold_p=0.05):
        self.threshold_p       = threshold_p
        self.selected_indices_ = None
        self.selected_names_   = None
        self.removal_log_      = []
        self.coefficients_     = None
        self.intercept_        = 0.0

    def _p_values(self, X_b, y):
        from scipy.stats import t as t_dist
        n, k = X_b.shape
        beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        resid  = y - X_b @ beta
        sigma2 = float(resid @ resid) / max(n - k, 1)
        try:
            cov  = sigma2 * np.linalg.pinv(X_b.T @ X_b)
            se   = np.sqrt(np.abs(np.diag(cov)))
            t    = beta / (se + 1e-12)
            pv   = 2 * t_dist.sf(np.abs(t), df=max(n - k, 1))
        except Exception:
            pv = np.zeros(k)
        return pv[1:], beta   # skip intercept p-value

    def fit(self, X, y, feature_names=None):
        X, y  = self._validate_X_y(X, y)
        n, p  = X.shape
        names = list(feature_names or [f"X{i}" for i in range(p)])
        remaining = list(range(p))
        self.removal_log_ = []

        while True:
            X_b      = np.column_stack([np.ones(n), X[:, remaining]])
            pv, beta = self._p_values(X_b, y)
            max_idx  = int(np.argmax(pv))
            max_p    = pv[max_idx]
            if max_p > self.threshold_p:
                self.removal_log_.append(
                    {"removed": names[max_idx], "p_value": max_p})
                names.pop(max_idx)
                remaining.pop(max_idx)
            else:
                break

        self.selected_indices_ = remaining
        self.selected_names_   = names
        X_b = np.column_stack([np.ones(n), X[:, remaining]])
        beta, *_ = np.linalg.lstsq(X_b, y, rcond=None)
        self.intercept_    = float(beta[0])
        self.coefficients_ = beta[1:]
        return self

    def predict(self, X):
        self._check_fitted("selected_indices_")
        X = self._validate_X_y(X)
        return self.intercept_ + X[:, self.selected_indices_] @ self.coefficients_

    def summary(self):
        print("=== Backward Elimination ===")
        for r in self.removal_log_:
            print(f"  Removed '{r['removed']}'  (p={r['p_value']:.4f})")
        print(f"  Features kept: {self.selected_names_}")
