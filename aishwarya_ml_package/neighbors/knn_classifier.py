"""KNN Regressor and Classifier — City Block distance, Min-Max normalization."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from aishwarya_ml_package.base       import BaseEstimator
from aishwarya_ml_package.neighbors.distance import manhattan, euclidean, cosine
from aishwarya_ml_package.neighbors.voting   import majority_vote, weighted_vote


class _KNNBase(BaseEstimator):
    """Shared fit / normalization logic for KNN."""

    DISTANCE_FNS = {"manhattan": manhattan,
                    "euclidean": euclidean,
                    "cosine"   : cosine}

    def __init__(self, k=5, distance="manhattan", normalize=True):
        self.k         = k
        self.distance  = distance
        self.normalize = normalize
        self.X_train_  = None
        self.y_train_  = None
        self._X_min    = None
        self._X_max    = None
        self.coefficients_ = None   # dummy for _check_fitted

    def _norm(self, X, fit=False):
        if not self.normalize:
            return X
        if fit:
            self._X_min = X.min(axis=0)
            self._X_max = X.max(axis=0)
        rng = self._X_max - self._X_min
        rng[rng == 0] = 1
        return (X - self._X_min) / rng

    def fit(self, X, y):
        X, y = self._validate_X_y(X, y)
        self.X_train_      = self._norm(X, fit=True)
        self.y_train_      = y.copy()
        self.coefficients_ = True   # mark as fitted
        return self

    def _get_neighbors(self, x_norm):
        dist_fn = self.DISTANCE_FNS.get(self.distance, manhattan)
        dists   = dist_fn(x_norm, self.X_train_)
        idx     = np.argsort(dists)[:self.k]
        return idx, dists[idx]

    def summary(self):
        print("=" * 42)
        print(f"  {self.__class__.__name__}")
        print("=" * 42)
        print(f"  K              : {self.k}")
        print(f"  Distance       : {self.distance}")
        print(f"  Normalization  : {self.normalize}")
        if self.X_train_ is not None:
            print(f"  Train samples  : {len(self.X_train_)}")
        print("=" * 42)


class KNNRegressor(_KNNBase):
    """
    KNN Regressor.
    Prediction = mean of K nearest neighbors' target values.
    Default distance: Manhattan (City Block).
    """
    def __init__(self, k=5, distance="manhattan", normalize=True):
        super().__init__(k=k, distance=distance, normalize=normalize)

    def predict(self, X):
        self._check_fitted()
        X     = self._validate_X_y(X)
        X_n   = self._norm(X)
        preds = []
        for point in X_n:
            idx, _ = self._get_neighbors(point)
            preds.append(float(np.mean(self.y_train_[idx])))
        return np.array(preds)

    def get_neighbors(self, x):
        """Return neighbor indices, distances, and labels for one sample."""
        self._check_fitted()
        x   = np.array(x, dtype=np.float64).reshape(1, -1)
        x_n = self._norm(x)[0]
        idx, dists = self._get_neighbors(x_n)
        return {"indices": idx.tolist(),
                "distances": dists.tolist(),
                "targets": self.y_train_[idx].tolist()}


class KNNClassifier(_KNNBase):
    """
    KNN Classifier.
    Prediction = majority vote among K nearest neighbors.
    Supports weighted voting (weight = 1/distance).
    """
    def __init__(self, k=5, distance="manhattan", normalize=True,
                 weights="uniform"):
        super().__init__(k=k, distance=distance, normalize=normalize)
        self.weights = weights   # "uniform" or "distance"

    def predict(self, X):
        self._check_fitted()
        X   = self._validate_X_y(X)
        X_n = self._norm(X)
        preds = []
        for point in X_n:
            idx, dists = self._get_neighbors(point)
            labels = self.y_train_[idx]
            if self.weights == "distance":
                pred = weighted_vote(labels, dists)
            else:
                pred = majority_vote(labels)
            preds.append(pred)
        return np.array(preds)

    def predict_proba(self, X):
        """Return class probabilities (vote fraction per class)."""
        self._check_fitted()
        X      = self._validate_X_y(X)
        X_n    = self._norm(X)
        classes= np.unique(self.y_train_)
        probas = []
        for point in X_n:
            idx, _ = self._get_neighbors(point)
            labels = self.y_train_[idx]
            row    = [np.mean(labels == c) for c in classes]
            probas.append(row)
        return np.array(probas)
