"""
Neural Network Regressor — from scratch.

Algorithm (from notes):
  1. Get data
  2. Normalize data
  3. Assign random weights (He init)
  4. Feed forward  → compute f(x)
  5. Find delta = f(x) - f(h)   [predicted - actual]
  6. If delta < epsilon → Stop
     Else → Wn = W0 + eta * delta * x
  7. Go to step 4
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base                      import BaseEstimator
from aishwarya_ml_package.neural_network.activations import ACTIVATIONS
from aishwarya_ml_package.neural_network.losses      import mse as mse_fn


class NeuralNetwork(BaseEstimator):
    """
    Feedforward Neural Network for regression.

    Parameters
    ----------
    hidden_layers  : tuple  — neurons per hidden layer, e.g. (16, 8)
    activation     : str    — hidden activation: 'relu', 'sigmoid', 'tanh'
    learning_rate  : float  — eta (η)
    epochs         : int    — max iterations
    epsilon        : float  — convergence threshold
    random_state   : int    — reproducibility seed
    verbose        : bool   — print progress
    """
    def __init__(self, hidden_layers=(16, 8), activation="relu",
                 learning_rate=0.01, epochs=2000,
                 epsilon=1e-6, random_state=42, verbose=True):
        self.hidden_layers  = hidden_layers
        self.activation     = activation
        self.learning_rate  = learning_rate
        self.epochs         = epochs
        self.epsilon        = epsilon
        self.random_state   = random_state
        self.verbose        = verbose
        self.weights_       = []
        self.biases_        = []
        self.loss_history_  = []
        self.coefficients_  = None   # set after fit for _check_fitted
        self._X_min = self._X_max = None
        self._y_min = self._y_max = None

    # ── Normalization ─────────────────────────────────────────────────────────
    def _norm_X(self, X, fit=False):
        if fit:
            self._X_min = X.min(axis=0)
            self._X_max = X.max(axis=0)
        rng = self._X_max - self._X_min
        rng[rng == 0] = 1
        return (X - self._X_min) / rng

    def _norm_y(self, y, fit=False):
        if fit:
            self._y_min = float(y.min())
            self._y_max = float(y.max())
        rng = self._y_max - self._y_min or 1.0
        return (y - self._y_min) / rng

    def _denorm_y(self, yn):
        return yn * (self._y_max - self._y_min) + self._y_min

    # ── Step 3: Initialize weights ─────────────────────────────────────────────
    def _init_weights(self, n_in):
        np.random.seed(self.random_state)
        sizes = [n_in] + list(self.hidden_layers) + [1]
        self.weights_ = []
        self.biases_  = []
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.weights_.append(W)
            self.biases_.append(b)

    # ── Step 4: Feed forward ───────────────────────────────────────────────────
    def _forward(self, X):
        act_fn, _ = ACTIVATIONS.get(self.activation, ACTIVATIONS["relu"])
        self._acts = [X]
        self._zs   = []
        cur = X
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z   = cur @ W + b
            self._zs.append(z)
            cur = act_fn(z) if i < len(self.weights_) - 1 else z  # linear output
            self._acts.append(cur)
        return cur

    # ── Steps 5-6: Backprop ────────────────────────────────────────────────────
    def _backward(self, y):
        _, act_d = ACTIVATIONS.get(self.activation, ACTIVATIONS["relu"])
        n     = y.shape[0]
        delta = self._acts[-1] - y.reshape(-1, 1)   # Step 5: delta = f(x) - f(h)
        for i in reversed(range(len(self.weights_))):
            dW     = (self._acts[i].T @ delta) / n
            db     = np.mean(delta, axis=0, keepdims=True)
            W_orig = self.weights_[i].copy()
            self.weights_[i] -= self.learning_rate * dW   # Step 6: Wn = W0 + η·Δx
            self.biases_[i]  -= self.learning_rate * db
            if i > 0:
                delta = (delta @ W_orig.T) * act_d(self._zs[i - 1])

    # ── Main training loop ─────────────────────────────────────────────────────
    def fit(self, X, y):
        X, y = self._validate_X_y(X, y)
        X_n  = self._norm_X(X, fit=True)
        y_n  = self._norm_y(y, fit=True)
        self._init_weights(X_n.shape[1])
        self.loss_history_ = []

        for epoch in range(self.epochs):
            out   = self._forward(X_n)              # Step 4
            loss  = mse_fn(y_n, out.flatten())
            self.loss_history_.append(loss)

            if loss < self.epsilon:                 # Step 6: if delta < ε → Stop
                if self.verbose:
                    print(f"  Converged at epoch {epoch+1}  (MSE={loss:.2e})")
                break

            self._backward(y_n)                     # Step 6: update weights

            if self.verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1:>5}/{self.epochs}  |  MSE={loss:.6f}")

        self.coefficients_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        X   = self._validate_X_y(X)
        X_n = self._norm_X(X)
        out = self._forward(X_n).flatten()
        return self._denorm_y(out)

    def get_loss_history(self):
        return self.loss_history_

    def plot_loss(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required.")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        epochs = range(1, len(self.loss_history_) + 1)
        axes[0].plot(epochs, self.loss_history_, color="#e74c3c", linewidth=1.5)
        axes[0].set_title("Training Loss (MSE)", fontweight="bold")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")

        axes[1].plot(epochs, np.log(np.array(self.loss_history_) + 1e-12),
                     color="#2980b9", linewidth=1.5)
        axes[1].set_title("Training Loss — Log Scale", fontweight="bold")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("log(MSE)")

        plt.suptitle("Neural Network — Convergence", fontweight="bold")
        plt.tight_layout(); plt.show()

    def summary(self):
        print("=" * 48)
        print("  Neural Network  (from scratch)")
        print("=" * 48)
        print(f"  Architecture   : {self._get_arch_str()}")
        print(f"  Activation     : {self.activation}")
        print(f"  Learning rate  : {self.learning_rate}")
        print(f"  Epochs (max)   : {self.epochs}")
        print(f"  Epsilon (ε)    : {self.epsilon}")
        if self.loss_history_:
            print(f"  Epochs run     : {len(self.loss_history_)}")
            print(f"  Final MSE      : {self.loss_history_[-1]:.8f}")
        print("=" * 48)

    def _get_arch_str(self):
        if not self.weights_:
            return "not built"
        parts = [str(self.weights_[0].shape[0])]
        for W in self.weights_:
            parts.append(str(W.shape[1]))
        return " → ".join(parts)
