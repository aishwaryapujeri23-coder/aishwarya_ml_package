"""Utility helpers used across ml_package."""
import numpy as np


def check_array(X, ensure_2d=True):
    """Convert X to float64 ndarray; optionally enforce 2D."""
    X = np.array(X, dtype=np.float64)
    if ensure_2d and X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def check_X_y(X, y):
    """Convert and validate both X and y."""
    X = check_array(X)
    y = np.array(y, dtype=np.float64).ravel()
    if len(X) != len(y):
        raise ValueError(
            f"X has {len(X)} samples but y has {len(y)} samples."
        )
    return X, y


def column_names(X, prefix="X"):
    """Return default column names if none given."""
    n = X.shape[1] if X.ndim == 2 else 1
    return [f"{prefix}{i}" for i in range(n)]
