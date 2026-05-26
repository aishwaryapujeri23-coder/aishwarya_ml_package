"""Classification metrics."""
import numpy as np


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix. Returns ndarray."""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, average="macro"):
    """Precision per class, then averaged."""
    classes = np.unique(y_true)
    scores  = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(scores))


def recall(y_true, y_pred, average="macro"):
    """Recall per class, then averaged."""
    classes = np.unique(y_true)
    scores  = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(scores))


def f1(y_true, y_pred, average="macro"):
    """F1 = 2 * precision * recall / (precision + recall)"""
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def classification_report(y_true, y_pred):
    """Print per-class precision, recall, F1."""
    classes = np.unique(y_true)
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 44)
    for c in classes:
        tp  = np.sum((y_pred == c) & (y_true == c))
        fp  = np.sum((y_pred == c) & (y_true != c))
        fn  = np.sum((y_pred != c) & (y_true == c))
        sup = np.sum(y_true == c)
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f   = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        print(f"  {str(c):<10} {p:>10.4f} {r:>10.4f} {f:>10.4f} {sup:>10}")
    print()
    print(f"  Accuracy : {accuracy(y_true, y_pred):.4f}")
    print(f"  Macro F1 : {f1(y_true, y_pred):.4f}")
