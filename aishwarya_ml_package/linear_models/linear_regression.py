"""Linear Regression via Normal Equation + OLS summary."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class LinearRegression(BaseEstimator):
    """
    Ordinary Least Squares Linear Regression — Normal Equation.
    beta = (X'X)^-1 X'y  solved via lstsq for numerical stability.

    Includes OLS summary with p-values, F-stat, R², Adj-R².
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept  = fit_intercept
        self.coefficients_  = None
        self.intercept_     = 0.0
        self.residuals_     = None
        self._ols_stats     = {}

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y):
        X, y = self._validate_X_y(X, y)
        self._n, self._p = X.shape
        X_b = self._add_intercept(X)

        # Normal Equation
        beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

        if self.fit_intercept:
            self.intercept_    = float(beta[0])
            self.coefficients_ = beta[1:]
        else:
            self.coefficients_ = beta

        self.residuals_ = y - X_b @ beta
        self._compute_ols_stats(X_b, y, beta)
        return self

    def _compute_ols_stats(self, X_b, y, beta):
        """Compute OLS statistics: p-values, F-stat, R², Adj-R²."""
        n, k  = X_b.shape
        y_hat = X_b @ beta
        resid = y - y_hat
        SSR   = float(resid @ resid)
        SST   = float(((y - y.mean()) ** 2).sum())
        SSE   = SST - SSR

        r2      = 1 - SSR / SST if SST > 0 else 0.0
        adj_r2  = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else 0.0
        sigma2  = SSR / max(n - k, 1)

        try:
            cov_beta = sigma2 * np.linalg.pinv(X_b.T @ X_b)
            se       = np.sqrt(np.abs(np.diag(cov_beta)))
            t_stats  = beta / (se + 1e-12)
            from scipy.stats import t as t_dist
            p_values = 2 * t_dist.sf(np.abs(t_stats), df=max(n - k, 1))
        except Exception:
            se = t_stats = p_values = np.zeros(k)

        f_stat = (SSE / (k - 1)) / (SSR / max(n - k, 1)) if n > k else 0.0

        self._ols_stats = {
            "r2": r2, "adj_r2": adj_r2, "sigma2": sigma2,
            "f_stat": f_stat, "se": se,
            "t_stats": t_stats, "p_values": p_values,
            "SSR": SSR, "SST": SST, "n": n, "k": k
        }

    def predict(self, X):
        self._check_fitted()
        X = self._validate_X_y(X)
        return self.intercept_ + X @ self.coefficients_

    def ols_summary(self, feature_names=None):
        """Print a full OLS-style regression summary."""
        s     = self._ols_stats
        names = (["const"] + list(feature_names)) if feature_names \
                else ["const"] + [f"X{i}" for i in range(len(self.coefficients_))]
        beta  = np.concatenate([[self.intercept_], self.coefficients_])

        print("=" * 65)
        print("  OLS Regression Results")
        print("=" * 65)
        print(f"  R²          : {s['r2']:.4f}")
        print(f"  Adj. R²     : {s['adj_r2']:.4f}")
        print(f"  F-Statistic : {s['f_stat']:.4f}")
        print(f"  Observations: {s['n']}   |   Variables: {s['k']}")
        print("-" * 65)
        print(f"  {'Variable':<20} {'Coef':>10} {'Std Err':>10} "
              f"{'t':>10} {'P>|t|':>10}")
        print("-" * 65)
        for name, b, se, t, p in zip(
                names, beta, s["se"], s["t_stats"], s["p_values"]):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else \
                  "*"   if p < 0.05  else "."  if p < 0.1  else ""
            print(f"  {name:<20} {b:>10.4f} {se:>10.4f} {t:>10.4f} {p:>10.4f} {sig}")
        print("=" * 65)
        print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")
        print("=" * 65)
