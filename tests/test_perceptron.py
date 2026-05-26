import numpy as np


class Perceptron:
    """
    Perceptron Regression Model

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for weight updates.

    n_iter : int, default=1000
        Maximum number of training iterations.

    epsilon : float, default=1e-6
        Minimum error threshold for stopping.

    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_iter=1000,
        epsilon=1e-6,
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

        self.weights = rng.uniform(
            low=0.0,
            high=1.0,
            size=n_features
        )

        self.bias = rng.uniform(
            low=0.0,
            high=1.0
        )

    def _feedforward(self, X):
        """
        Compute linear output.
        """

        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        """
        Train the perceptron model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input data.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Trained model.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1D array."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Number of samples in X and y must match."
            )

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

            # Numerical stability check
            if np.isnan(mse) or np.isinf(mse):
                raise ValueError(
                    "Training diverged. "
                    "Try smaller learning_rate."
                )

            self.loss_history.append(mse)

            # Step 6: Termination condition
            if mse < self.epsilon:
                print(
                    f"Training converged "
                    f"at epoch {epoch + 1}"
                )
                break

            # Step 7: Weight update
            dw = (
                self.learning_rate
                * np.dot(X.T, error)
                / n_samples
            )

            db = (
                self.learning_rate
                * np.mean(error)
            )

            self.weights += dw
            self.bias += db

        return self

    def predict(self, X):
        """
        Predict target values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Predicted values.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self._feedforward(X)

    def score(self, X, y):
        """
        Compute R² score.

        Parameters
        ----------
        X : ndarray
            Input data.

        y : ndarray
            True target values.

        Returns
        -------
        float
            R² score.
        """

        y = np.asarray(y, dtype=float)

        y_pred = self.predict(X)

        ss_total = np.sum(
            (y - np.mean(y)) ** 2
        )

        ss_residual = np.sum(
            (y - y_pred) ** 2
        )

        if ss_total == 0:
            return 0.0

        return 1 - (ss_residual / ss_total)

    def get_params(self):
        """
        Return model parameters.
        """

        return {
            "learning_rate": self.learning_rate,
            "n_iter": self.n_iter,
            "epsilon": self.epsilon,
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