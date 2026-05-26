"""Tests for linear_models module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from aishwarya_ml_package.linear_models import LinearRegression, Ridge, Lasso
from aishwarya_ml_package.linear_models import ForwardSelection, BackwardElimination


np.random.seed(42)
X = np.random.randn(100, 3)
y = 3*X[:,0] + 2*X[:,1] - X[:,2] + np.random.randn(100)*0.1


class TestLinearRegression:
    def test_fit_predict_shape(self):
        m = LinearRegression(); m.fit(X, y)
        assert m.predict(X).shape == (100,)

    def test_r2_high(self):
        from aishwarya_ml_package.metrics import r2
        m = LinearRegression(); m.fit(X, y)
        assert r2(y, m.predict(X)) > 0.99

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            LinearRegression().predict(X)

    def test_coefficients_close(self):
        m = LinearRegression(); m.fit(X, y)
        np.testing.assert_allclose(m.coefficients_, [3, 2, -1], atol=0.1)

    def test_ols_summary_runs(self, capsys):
        m = LinearRegression(); m.fit(X, y)
        m.ols_summary(["A","B","C"])
        out = capsys.readouterr().out
        assert "OLS" in out


class TestRidge:
    def test_fit_predict(self):
        m = Ridge(alpha=1.0); m.fit(X, y)
        assert m.predict(X).shape == (100,)

    def test_regularization_reduces_coef(self):
        lr = LinearRegression(); lr.fit(X, y)
        r  = Ridge(alpha=100.0); r.fit(X, y)
        assert np.abs(r.coefficients_).sum() < np.abs(lr.coefficients_).sum()


class TestLasso:
    def test_fit_predict(self):
        m = Lasso(alpha=0.01); m.fit(X, y)
        assert m.predict(X).shape == (100,)


class TestForwardSelection:
    def test_selects_features(self):
        m = ForwardSelection(); m.fit(X, y)
        assert len(m.selected_indices_) > 0

    def test_predict_shape(self):
        m = ForwardSelection(); m.fit(X, y)
        assert m.predict(X).shape == (100,)


class TestBackwardElimination:
    def test_keeps_significant(self):
        m = BackwardElimination(threshold_p=0.05); m.fit(X, y)
        assert len(m.selected_names_) >= 1
