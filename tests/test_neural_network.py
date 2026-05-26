"""Tests for NeuralNetwork."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from aishwarya_ml_package.neural_network import NeuralNetwork


np.random.seed(7)
X = np.random.randn(100, 3)
y = 2*X[:,0] - X[:,1] + np.random.randn(100)*0.3


class TestNeuralNetwork:
    def test_predict_shape(self):
        m = NeuralNetwork(hidden_layers=(8,), epochs=100, verbose=False)
        m.fit(X, y)
        assert m.predict(X).shape == (100,)

    def test_loss_decreases(self):
        m = NeuralNetwork(hidden_layers=(16, 8), epochs=300, verbose=False)
        m.fit(X, y)
        hist = m.get_loss_history()
        assert hist[-1] < hist[0]

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            NeuralNetwork().predict(X)

    def test_activation_options(self):
        for act in ["relu", "sigmoid", "tanh"]:
            m = NeuralNetwork(hidden_layers=(8,), activation=act,
                              epochs=50, verbose=False)
            m.fit(X, y)
            assert m.predict(X[:5]).shape == (5,)

    def test_summary_runs(self, capsys):
        m = NeuralNetwork(hidden_layers=(8,), epochs=50, verbose=False)
        m.fit(X, y)
        m.summary()
        out = capsys.readouterr().out
        assert "Neural Network" in out
