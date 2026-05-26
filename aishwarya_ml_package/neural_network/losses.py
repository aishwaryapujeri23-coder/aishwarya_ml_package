"""Loss functions for neural network training."""
import numpy as np


def mse(y_true, y_pred):
    """Mean Squared Error — regression."""
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error — regression."""
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def binary_crossentropy(y_true, y_pred):
    """Binary cross-entropy — binary classification."""
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return float(-np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
