"""Check linear regression assumptions: normality, multicollinearity, heteroscedasticity."""
import numpy as np


def check_assumptions(X, y, y_pred, feature_names=None):
    """
    Run all OLS assumption checks and print a report.

    Checks
    ------
    1. Normality of residuals   (Shapiro-Wilk)
    2. Multicollinearity        (VIF)
    3. Heteroscedasticity       (Breusch-Pagan approximation)

    Parameters
    ----------
    X       : ndarray (n, p)  — feature matrix
    y       : ndarray (n,)    — actual target
    y_pred  : ndarray (n,)    — predicted target
    feature_names : list of str (optional)
    """
    from scipy.stats import shapiro
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    resid  = y - y_pred
    names  = feature_names or [f"X{i}" for i in range(X.shape[1])]

    print("=" * 55)
    print("  OLS Assumption Checks")
    print("=" * 55)

    # 1. Normality
    stat, p = shapiro(resid)
    print(f"\n  1. Normality of Residuals (Shapiro-Wilk)")
    print(f"     W={stat:.4f}, p={p:.4f}  → "
          f"{'✅ Normal' if p > 0.05 else '⚠️  Not Normal'} (p>0.05 = normal)")

    # 2. VIF
    print(f"\n  2. Multicollinearity (VIF)")
    print(f"     {'Feature':<18} {'VIF':>8}")
    print("     " + "-" * 28)
    X_b = np.column_stack([np.ones(len(X)), X])
    for j in range(1, X_b.shape[1]):
        y_j = X_b[:, j]
        X_j = np.delete(X_b, j, axis=1)
        beta, *_ = np.linalg.lstsq(X_j, y_j, rcond=None)
        y_hat_j  = X_j @ beta
        ss_tot   = ((y_j - y_j.mean()) ** 2).sum()
        ss_res   = ((y_j - y_hat_j) ** 2).sum()
        r2_j     = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif      = 1 / (1 - r2_j) if r2_j < 1 else float("inf")
        flag     = "✅" if vif < 5 else "⚠️ " if vif < 10 else "❌"
        print(f"     {names[j-1]:<18} {vif:>8.2f} {flag}")
    print("     VIF < 5 = OK  |  5-10 = Moderate  |  >10 = High")

    # 3. Heteroscedasticity (Breusch-Pagan approximation)
    print(f"\n  3. Heteroscedasticity (Breusch-Pagan approx.)")
    resid2 = resid ** 2
    X_b2   = np.column_stack([np.ones(len(X)), X])
    beta2, *_ = np.linalg.lstsq(X_b2, resid2, rcond=None)
    fitted2   = X_b2 @ beta2
    ss_res2   = ((resid2 - fitted2) ** 2).sum()
    ss_tot2   = ((resid2 - resid2.mean()) ** 2).sum()
    r2_bp     = 1 - ss_res2 / ss_tot2 if ss_tot2 > 0 else 0
    lm        = len(X) * r2_bp
    from scipy.stats import chi2
    p_bp = float(chi2.sf(lm, df=X.shape[1]))
    print(f"     LM={lm:.4f}, p={p_bp:.4f}  → "
          f"{'✅ Homoscedastic' if p_bp > 0.05 else '⚠️  Heteroscedastic'} (p>0.05 = OK)")
    print("=" * 55)
