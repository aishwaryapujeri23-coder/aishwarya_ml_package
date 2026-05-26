"""PCA via Eigen Decomposition — from scratch."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base import BaseEstimator


class PCA(BaseEstimator):
    """
    Principal Component Analysis from scratch.
    Steps: Center → Covariance → Eigen decomp → Sort → Project.

    Usage
    -----
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    pca.summary()
    pca.plot_variance()
    """
    def __init__(self, n_components=None):
        self.n_components             = n_components
        self.components_              = None
        self.explained_variance_      = None
        self.explained_variance_ratio_= None
        self.mean_                    = None

    def fit(self, X):
        X    = np.array(X, dtype=np.float64)
        n, p = X.shape
        n_comp = min(self.n_components or p, p)

        self.mean_ = X.mean(axis=0)
        Xc  = X - self.mean_
        cov = (Xc.T @ Xc) / (n - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_          = eigenvectors[:, :n_comp].T
        self.explained_variance_  = eigenvalues[:n_comp]
        total = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else np.zeros(n_comp))
        self._all_eigenvalues = eigenvalues
        return self

    def transform(self, X):
        self._check_fitted("components_")
        X = np.array(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        self._check_fitted("components_")
        return np.array(Z, dtype=np.float64) @ self.components_ + self.mean_

    def explained_variance_dict(self):
        return {f"PC{i+1}": round(float(r), 6)
                for i, r in enumerate(self.explained_variance_ratio_)}

    def plot_variance(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib needed for plot_variance.")
        ratios     = self.explained_variance_ratio_
        cumulative = np.cumsum(ratios)
        components = [f"PC{i+1}" for i in range(len(ratios))]
        created    = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(components, ratios * 100, color="#3498db",
               edgecolor="white", alpha=0.85, label="Individual %")
        ax2 = ax.twinx()
        ax2.plot(components, cumulative * 100, "s-", color="#e74c3c",
                 linewidth=2, markersize=8, label="Cumulative %")
        ax2.axhline(90, color="green", linestyle="--", linewidth=1.2)
        ax2.set_ylabel("Cumulative %"); ax2.set_ylim(0, 105)
        ax.set_title("PCA Scree Plot", fontweight="bold")
        ax.set_xlabel("Component"); ax.set_ylabel("Explained Variance %")
        if created:
            plt.tight_layout(); plt.show()

    def summary(self):
        self._check_fitted("components_")
        print("=" * 52)
        print("  PCA — Principal Component Analysis")
        print("=" * 52)
        print(f"  Components : {len(self.explained_variance_ratio_)}")
        print(f"  Total var  : {self.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"\n  {'PC':<6} {'Eigenvalue':>12} {'Var %':>9} {'Cumul %':>10}")
        print("  " + "-" * 40)
        cum = 0.0
        for i, (ev, r) in enumerate(zip(
                self.explained_variance_, self.explained_variance_ratio_)):
            cum += r * 100
            print(f"  PC{i+1:<4} {ev:>12.4f} {r*100:>8.2f}% {cum:>9.2f}%")
        print("=" * 52)
