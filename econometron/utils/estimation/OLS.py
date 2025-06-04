from scipy.stats import norm
import numpy as np

def ols_estimator(X, Y):
    """
    Perform OLS estimation with standard errors.
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    fitted = X @ beta
    residuals = Y - fitted
    T, K = Y.shape
    resid_cov = np.cov(residuals.T)
    XTX_inv = np.linalg.inv(X.T @ X)
    se = np.zeros_like(beta)
    for k in range(K):
        se[:, k] = np.sqrt(resid_cov[k, k] * np.diag(XTX_inv))
    z_values = beta / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))
    return beta, residuals, se, z_values, p_values