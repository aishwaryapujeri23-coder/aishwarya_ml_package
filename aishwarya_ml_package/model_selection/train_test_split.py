"""Train-test split and K-Fold cross-validation."""
import numpy as np


def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True):
    """
    Split arrays into train and test sets.

    Parameters
    ----------
    *arrays     : X, y  (or just X)
    test_size   : float — fraction for test (default 0.2 = 20%)
    random_state: int   — seed for reproducibility
    shuffle     : bool  — shuffle before splitting

    Returns
    -------
    Pairs: (X_train, X_test) or (X_train, X_test, y_train, y_test)

    Example
    -------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(arrays[0])
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    n_test  = int(np.ceil(n * test_size))
    n_train = n - n_test
    idx_train = idx[:n_train]
    idx_test  = idx[n_train:]

    result = []
    for arr in arrays:
        arr = np.array(arr)
        result.append(arr[idx_train])
        result.append(arr[idx_test])
    return result


class KFoldCV:
    """
    K-Fold Cross-Validation.

    Usage
    -----
    kf = KFoldCV(n_splits=5, random_state=42)
    for X_train, X_val, y_train, y_val in kf.split(X, y):
        model.fit(X_train, y_train)
        score = r2(y_val, model.predict(X_val))
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits    = n_splits
        self.shuffle     = shuffle
        self.random_state= random_state

    def split(self, X, y=None):
        X   = np.array(X)
        n   = len(X)
        idx = np.arange(n)
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(idx)

        folds = np.array_split(idx, self.n_splits)
        for i in range(self.n_splits):
            val_idx   = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            if y is not None:
                y = np.array(y)
                yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]
            else:
                yield X[train_idx], X[val_idx]

    def cross_val_score(self, model, X, y, metric_fn):
        """
        Run K-fold CV and return list of scores.

        Parameters
        ----------
        model     : any ml_package model with fit() and predict()
        X, y      : data
        metric_fn : e.g. from ml_package.metrics.regression import r2
        """
        scores = []
        for X_tr, X_val, y_tr, y_val in self.split(X, y):
            import copy
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            scores.append(metric_fn(y_val, m.predict(X_val)))
        print(f"  CV Scores : {[round(s,4) for s in scores]}")
        print(f"  Mean      : {np.mean(scores):.4f}")
        print(f"  Std       : {np.std(scores):.4f}")
        return scores
