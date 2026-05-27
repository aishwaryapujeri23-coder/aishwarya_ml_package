"""Regression metrics."""
import numpy as np


def r2(y_true, y_pred):
    """R-squared (coefficient of determination). Best=1.0"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def adjusted_r2(y_true, y_pred, n_predictors):
    """Adjusted R-squared for regression models."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    n = len(y_true)
    if n <= n_predictors + 1:
        return 0.0
    r2_val = r2(y_true, y_pred)
    return float(1 - (1 - r2_val) * (n - 1) / (n - n_predictors - 1))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)


def all_metrics(y_true, y_pred, label="Model", n_predictors=None):
    """Print and return all regression metrics."""
    results = {
        "R2": r2(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    if n_predictors is not None:
        results["AdjR2"] = adjusted_r2(y_true, y_pred, n_predictors)

    print(f"\n  Regression Metrics — {label}")
    print(f"  {'R²':<8}: {results['R2']:.4f}  ({results['R2']*100:.2f}% variance explained)")
    if "AdjR2" in results:
        print(f"  {'Adj. R²':<8}: {results['AdjR2']:.4f}")
    print(f"  {'RMSE':<8}: {results['RMSE']:.4f}")
    print(f"  {'MAE':<8}: {results['MAE']:.4f}")
    print(f"  {'MAPE':<8}: {results['MAPE']:.2f}%")
    return results
