"""Lasso Regression — L1 Regularization via Coordinate Descent."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class Lasso(BaseEstimator):
    """
    Lasso Regression: can shrink coefficients to exactly 0.
    Loss = RSS + alpha * sum(|beta_j|)
    Optimized via Coordinate Descent + Soft-Thresholding.
    """
    def __init__(self, alpha=0.01, max_iter=2000, tol=1e-6,
                 fit_intercept=True):
        self.alpha         = alpha
        self.max_iter      = max_iter
        self.tol           = tol
        self.fit_intercept = fit_intercept
        self.coefficients_ = None
        self.intercept_    = 0.0

    @staticmethod
    def _soft_threshold(rho, alpha):
        if rho >  alpha: return rho - alpha
        if rho < -alpha: return rho + alpha
        return 0.0

    def fit(self, X, y):
        X, y = self._validate_X_y(X, y)
        n, p = X.shape
        beta = np.zeros(p)
        intercept = float(y.mean()) if self.fit_intercept else 0.0

        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j    = y - intercept - X @ beta + X[:, j] * beta[j]
                rho    = float(X[:, j] @ r_j) / n
                beta[j] = self._soft_threshold(rho, self.alpha)
            if self.fit_intercept:
                intercept = float(np.mean(y - X @ beta))
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        self.coefficients_ = beta
        self.intercept_    = intercept
        return self

    def predict(self, X):
        self._check_fitted()
        X = self._validate_X_y(X)
        return self.intercept_ + X @ self.coefficients_
