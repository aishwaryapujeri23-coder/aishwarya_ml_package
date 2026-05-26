"""OLS diagnostics — residual plots and statistics."""
import numpy as np


class OLSSummary:
    """
    Compute and display OLS diagnostics from a fitted LinearRegression.

    Usage
    -----
    diag = OLSSummary(model, X_test, y_test)
    diag.print_summary()
    diag.plot_residuals()
    """
    def __init__(self, model, X, y):
        self.model  = model
        self.X      = np.array(X, dtype=np.float64)
        self.y      = np.array(y, dtype=np.float64)
        self.y_pred = model.predict(X)
        self.resid  = self.y - self.y_pred

    def print_summary(self):
        resid = self.resid
        print("=== OLS Diagnostics ===")
        print(f"  Mean residual    : {resid.mean():.4f}")
        print(f"  Std  residual    : {resid.std():.4f}")
        print(f"  Min  residual    : {resid.min():.4f}")
        print(f"  Max  residual    : {resid.max():.4f}")
        from scipy.stats import shapiro, jarque_bera
        sw_w, sw_p       = shapiro(resid)
        jb_s, jb_p, *_  = jarque_bera(resid)
        print(f"  Shapiro-Wilk     : W={sw_w:.4f}, p={sw_p:.4f}")
        print(f"  Jarque-Bera      : stat={jb_s:.4f}, p={jb_p:.4f}")

    def plot_residuals(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plot_residuals.")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].scatter(self.y_pred, self.resid, alpha=0.6,
                        color="#3498db", edgecolor="white")
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title("Residuals vs Fitted", fontweight="bold")
        axes[0].set_xlabel("Fitted Values"); axes[0].set_ylabel("Residuals")

        axes[1].hist(self.resid, bins=25, color="#2ecc71", edgecolor="white")
        axes[1].set_title("Residual Distribution", fontweight="bold")

        from scipy.stats import probplot
        probplot(self.resid, plot=axes[2])
        axes[2].set_title("Q-Q Plot", fontweight="bold")

        plt.tight_layout()
        plt.show()
