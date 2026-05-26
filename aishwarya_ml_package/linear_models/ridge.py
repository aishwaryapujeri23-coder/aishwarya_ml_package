"""Ridge Regression — L2 Regularization."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class Ridge(BaseEstimator):
    """
    Ridge Regression: penalizes large coefficients.
    Loss = RSS + alpha * sum(beta_j^2)
    Closed-form: beta = (X'X + alpha*I)^-1 X'y
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha         = alpha
        self.fit_intercept = fit_intercept
        self.coefficients_ = None
        self.intercept_    = 0.0

    def fit(self, X, y):
        X, y = self._validate_X_y(X, y)
        if self.fit_intercept:
            X_b = np.column_stack([np.ones(len(X)), X])
        else:
            X_b = X
        n, k  = X_b.shape
        I     = np.eye(k)
        if self.fit_intercept:
            I[0, 0] = 0   # don't regularize intercept
        A    = X_b.T @ X_b + self.alpha * I
        beta, _, _, _ = np.linalg.lstsq(A, X_b.T @ y, rcond=None)
        if self.fit_intercept:
            self.intercept_    = float(beta[0])
            self.coefficients_ = beta[1:]
        else:
            self.coefficients_ = beta
        return self

    def predict(self, X):
        self._check_fitted()
        X = self._validate_X_y(X)
        return self.intercept_ + X @ self.coefficients_
