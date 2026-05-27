import numpy as np

from aishwarya_ml_package.metrics import adjusted_r2, all_metrics, r2


def test_adjusted_r2_computation():
    y_true = [3.0, 5.0, 2.0, 7.0]
    y_pred = [2.8, 4.9, 2.2, 7.1]
    expected = 1 - (1 - r2(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - 2)
    assert np.isclose(adjusted_r2(y_true, y_pred, n_predictors=1), expected)


def test_all_metrics_includes_adjusted_r2():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.2, 3.8]
    results = all_metrics(y_true, y_pred, label="AdjR2 Test", n_predictors=1)
    assert "AdjR2" in results
    assert np.isclose(results["AdjR2"], adjusted_r2(y_true, y_pred, n_predictors=1))
