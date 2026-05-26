"""
Base class for all estimators in ml_package.
Every model inherits from this — just like sklearn's BaseEstimator.
"""
import numpy as np


class BaseEstimator:
    """
    Base class that every model in ml_package inherits from.
    Provides: get_params(), set_params(), __repr__()
    """

    def get_params(self):
        """Return constructor parameters as a dict."""
        import inspect
        sig    = inspect.signature(self.__class__.__init__)
        params = {}
        for name, _ in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
        return params

    def set_params(self, **params):
        """Set parameters. Returns self for chaining."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

    def _validate_X_y(self, X, y=None):
        """Convert to numpy float64 and do basic shape checks."""
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y is not None:
            y = np.array(y, dtype=np.float64).ravel()
            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X:{len(X)}, y:{len(y)}"
                )
            return X, y
        return X

    def _check_fitted(self, attr="coefficients_"):
        """Raise error if model has not been trained yet."""
        if not hasattr(self, attr) or getattr(self, attr) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. "
                f"Call fit(X, y) before predict()."
            )
