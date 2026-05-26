"""Visualization tools for PCA and model results."""
import numpy as np


def plot_pca_2d(X_reduced, y=None, title="PCA — 2D Projection"):
    """
    Scatter plot of first 2 principal components.

    Parameters
    ----------
    X_reduced : ndarray (n, ≥2)  — PCA-transformed data
    y         : array-like        — labels/colors (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    fig, ax = plt.subplots(figsize=(8, 6))
    if y is not None:
        classes = np.unique(y)
        colors  = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        for c, col in zip(classes, colors):
            mask = np.array(y) == c
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       label=str(c), color=col, alpha=0.7, edgecolor="white", s=70)
        ax.legend(title="Class")
    else:
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                   alpha=0.7, color="#3498db", edgecolor="white", s=70)

    ax.set_xlabel("PC1", fontweight="bold")
    ax.set_ylabel("PC2", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()


def plot_scree(pca, title="Scree Plot"):
    """
    Plot explained variance and cumulative variance from a fitted PCA object.

    Parameters
    ----------
    pca : ml_package.decomposition.PCA — fitted PCA object
    """
    pca.plot_variance()


def plot_biplot(pca, X_reduced, feature_names=None, scale=1.0):
    """
    PCA biplot: scatter of samples + arrows for original features.

    Parameters
    ----------
    pca           : fitted PCA
    X_reduced     : ndarray (n, 2) — PCA scores
    feature_names : list of str
    scale         : float — arrow length multiplier
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    components  = pca.components_   # (n_comp, n_features)
    n_features  = components.shape[1]
    names       = feature_names or [f"F{i}" for i in range(n_features)]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
               alpha=0.5, color="#3498db", edgecolor="white", s=50)

    for i in range(n_features):
        ax.annotate("", xy=(components[0, i]*scale, components[1, i]*scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
        ax.text(components[0, i]*scale*1.1, components[1, i]*scale*1.1,
                names[i], fontsize=11, color="#e74c3c", fontweight="bold")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("PC1", fontweight="bold")
    ax.set_ylabel("PC2", fontweight="bold")
    ax.set_title("PCA Biplot", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
