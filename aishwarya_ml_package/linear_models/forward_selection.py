"""Forward Feature Selection based on Adjusted R²."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class ForwardSelection(BaseEstimator):
    """
    Greedy forward feature selection.
    At each step: add the feature that maximises Adjusted R².
    Stop when no feature improves the score.
    """
    def __init__(self):
        self.selected_indices_ = None
        self.selected_names_   = None
        self.history_          = []
        self.coefficients_     = None
        self.intercept_        = 0.0

    @staticmethod
    def _adj_r2(X_b, y):
        n, k  = X_b.shape
        beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        res   = y - X_b @ beta
        SSR   = float(res @ res)
        SST   = float(((y - y.mean()) ** 2).sum())
        r2    = 1 - SSR / SST if SST > 0 else 0
        return 1 - (1 - r2) * (n - 1) / max(n - k, 1), beta

    def fit(self, X, y, feature_names=None):
        X, y   = self._validate_X_y(X, y)
        n, p   = X.shape
        names  = feature_names or [f"X{i}" for i in range(p)]
        remaining = list(range(p))
        selected  = []
        best_adj  = -np.inf
        self.history_ = []

        while remaining:
            best_feat = None
            for idx in remaining:
                cand  = selected + [idx]
                X_b   = np.column_stack([np.ones(n), X[:, cand]])
                adj, _= self._adj_r2(X_b, y)
                if adj > best_adj:
                    best_adj  = adj
                    best_feat = idx
            if best_feat is None:
                break
            selected.append(best_feat)
            remaining.remove(best_feat)
            self.history_.append({"feature": names[best_feat],
                                   "adj_r2": best_adj})

        self.selected_indices_ = selected
        self.selected_names_   = [names[i] for i in selected]
        X_b = np.column_stack([np.ones(n), X[:, selected]])
        _, beta, _, _ = np.linalg.lstsq(X_b, y, rcond=None), \
                        *([None]*2), None
        beta, *_ = np.linalg.lstsq(X_b, y, rcond=None)
        self.intercept_    = float(beta[0])
        self.coefficients_ = beta[1:]
        return self

    def predict(self, X):
        self._check_fitted("selected_indices_")
        X = self._validate_X_y(X)
        return self.intercept_ + X[:, self.selected_indices_] @ self.coefficients_

    def summary(self):
        print("=== Forward Selection ===")
        for i, h in enumerate(self.history_, 1):
            print(f"  Step {i}: Added '{h['feature']}'  →  Adj R²={h['adj_r2']:.4f}")
        print(f"  Final features: {self.selected_names_}")
