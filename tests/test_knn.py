"""Tests for KNN Regressor and Classifier."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from aishwarya_ml_package.neighbors import KNNRegressor, KNNClassifier


np.random.seed(1)
X_reg  = np.random.randn(100, 3)
y_reg  = 2*X_reg[:,0] - X_reg[:,1] + np.random.randn(100)*0.2
X_cls  = np.random.randn(80, 2)
y_cls  = (X_cls[:,0] + X_cls[:,1] > 0).astype(float)


class TestKNNRegressor:
    def test_predict_shape(self):
        m = KNNRegressor(k=5); m.fit(X_reg, y_reg)
        assert m.predict(X_reg[:10]).shape == (10,)

    def test_k1_memorizes(self):
        m = KNNRegressor(k=1); m.fit(X_reg, y_reg)
        preds = m.predict(X_reg)
        np.testing.assert_allclose(preds, y_reg, atol=1e-8)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            KNNRegressor(k=3).predict(X_reg)

    def test_get_neighbors_length(self):
        m = KNNRegressor(k=5); m.fit(X_reg, y_reg)
        nb = m.get_neighbors(X_reg[0])
        assert len(nb["indices"]) == 5

    def test_distance_options(self):
        for dist in ["manhattan", "euclidean", "cosine"]:
            m = KNNRegressor(k=3, distance=dist); m.fit(X_reg, y_reg)
            assert m.predict(X_reg[:5]).shape == (5,)


class TestKNNClassifier:
    def test_predict_shape(self):
        m = KNNClassifier(k=5); m.fit(X_cls, y_cls)
        assert m.predict(X_cls).shape == (80,)

    def test_predict_proba_shape(self):
        m = KNNClassifier(k=3); m.fit(X_cls, y_cls)
        p = m.predict_proba(X_cls[:10])
        assert p.shape == (10, 2)

    def test_proba_sums_to_1(self):
        m = KNNClassifier(k=5); m.fit(X_cls, y_cls)
        p = m.predict_proba(X_cls)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(80), atol=1e-8)
