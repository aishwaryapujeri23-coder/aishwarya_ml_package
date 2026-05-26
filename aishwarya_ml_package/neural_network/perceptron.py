import numpy as np


class Perceptron:
    """
    Perceptron for Regression (Single Output Neuron)

    Parameters
    ----------
    learning_rate : float
        Learning rate for weight updates.

    n_iter : int
        Maximum number of iterations.

    epsilon : float
        Minimum error threshold for stopping training.

    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_iter=1000,
        epsilon=1e-4,
        random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.random_state = random_state

        self.weights = None
        self.bias = None

        self.loss_history = []

    def _initialize_weights(self, n_features):
        """
        Initialize weights randomly between 0 and 1.
        """
        rng = np.random.default_rng(self.random_state)

        self.weights = rng.random(n_features)
        self.bias = rng.random()

    def _feedforward(self, X):
        """
        Feedforward operation:
        sum(weights * inputs) + bias
        """
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        """
        Train the perceptron model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.

        y : ndarray of shape (n_samples,)
            Target values.
        """

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        # Step 2: Initialize layer
        self._initialize_weights(n_features)

        for epoch in range(self.n_iter):

            # Step 4: Feedforward
            y_pred = self._feedforward(X)

            # Step 5: Compute error
            error = y - y_pred

            # Mean Squared Error
            mse = np.mean(error ** 2)

            self.loss_history.append(mse)

            # Step 6: Termination condition
            if mse < self.epsilon:
                print(f"Training stopped at epoch {epoch}")
                break

            # Step 7: Backpropagation / weight update
            dw = self.learning_rate * np.dot(X.T, error) / n_samples
            db = self.learning_rate * np.mean(error)

            self.weights += dw
            self.bias += db

        return self

    def predict(self, X):
        """
        Predict output values.
        """

        X = np.array(X, dtype=float)

        return self._feedforward(X)

    def score(self, X, y):
        """
        Compute R² score.
        """

        y = np.array(y, dtype=float)
        y_pred = self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        return 1 - (ss_residual / ss_total)

    def get_params(self):
        """
        Return model parameters.
        """

        return {
            "weights": self.weights,
            "bias": self.bias
        }

    def __repr__(self):
        return (
            f"Perceptron("
            f"learning_rate={self.learning_rate}, "
            f"n_iter={self.n_iter}, "
            f"epsilon={self.epsilon})"
        )