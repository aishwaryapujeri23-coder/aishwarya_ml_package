"""Pipeline: chain preprocessing + model in one object."""
import numpy as np


class Pipeline:
    """
    Chain transformers + a final estimator into one object.
    Just like sklearn.pipeline.Pipeline.

    Usage
    -----
    from ml_package.pipeline    import Pipeline
    from ml_package.preprocessing import StandardScaler, MedianImputer
    from ml_package.linear_models import LinearRegression

    pipe = Pipeline([
        ("imputer", MedianImputer()),
        ("scaler",  StandardScaler()),
        ("model",   LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    pipe.summary()
    """

    def __init__(self, steps):
        """
        Parameters
        ----------
        steps : list of (name, estimator) tuples.
                Last step must have predict(). All others must have transform().
        """
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        for i, (name, step) in enumerate(self.steps[:-1]):
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"Step '{name}' (index {i}) must have a transform() method.")
        name, final = self.steps[-1]
        if not hasattr(final, "predict"):
            raise TypeError(
                f"Last step '{name}' must have a predict() method.")

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    def fit(self, X, y=None):
        """Fit all steps sequentially. Last step is fitted with (X_transformed, y)."""
        X_curr = np.array(X, dtype=np.float64)
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                X_curr = step.fit_transform(X_curr)
            else:
                step.fit(X_curr)
                X_curr = step.transform(X_curr)
        # Final estimator
        name, estimator = self.steps[-1]
        if y is not None:
            estimator.fit(X_curr, y)
        else:
            estimator.fit(X_curr)
        return self

    def transform(self, X):
        """Apply all transformer steps (not the final estimator)."""
        X_curr = np.array(X, dtype=np.float64)
        for name, step in self.steps[:-1]:
            X_curr = step.transform(X_curr)
        return X_curr

    def predict(self, X):
        """Transform X through all steps, then predict with final estimator."""
        X_curr = self.transform(X)
        return self._final_estimator.predict(X_curr)

    def fit_predict(self, X, y):
        """Fit and predict on the same data."""
        return self.fit(X, y).predict(X)

    def get_step(self, name):
        """Return a specific step by name."""
        for n, s in self.steps:
            if n == name:
                return s
        raise KeyError(f"Step '{name}' not found.")

    def summary(self):
        """Print the pipeline structure."""
        print("=" * 50)
        print("  Pipeline")
        print("=" * 50)
        for i, (name, step) in enumerate(self.steps):
            tag = "→ [MODEL]" if i == len(self.steps) - 1 else "→ [TRANSFORM]"
            print(f"  {i+1}. {name:<20} {step.__class__.__name__:<25} {tag}")
        print("=" * 50)
