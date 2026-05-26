"""Tests for PCA."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from aishwarya_ml_package.decomposition import PCA


np.random.seed(0)
X = np.random.randn(80, 5)


class TestPCA:
    def test_fit_transform_shape(self):
        pca = PCA(n_components=2)
        Z   = pca.fit_transform(X)
        assert Z.shape == (80, 2)

    def test_explained_variance_sums_to_1(self):
        pca = PCA()
        pca.fit(X)
        np.testing.assert_almost_equal(
            pca.explained_variance_ratio_.sum(), 1.0, decimal=5)

    def test_components_shape(self):
        pca = PCA(n_components=3)
        pca.fit(X)
        assert pca.components_.shape == (3, 5)

    def test_inverse_transform_close(self):
        pca = PCA(n_components=5)
        Z   = pca.fit_transform(X)
        X_r = pca.inverse_transform(Z)
        np.testing.assert_allclose(X, X_r, atol=1e-6)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            PCA(n_components=2).transform(X)

    def test_explained_variance_dict_keys(self):
        pca = PCA(n_components=2); pca.fit(X)
        d   = pca.explained_variance_dict()
        assert "PC1" in d and "PC2" in d
