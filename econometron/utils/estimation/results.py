# Statistics and Results Table
from scipy.stats import norm
import numpy as np

def compute_stats(params, log_lik, func, eps=1e-4):
    """
    Compute standard errors and p-values using numerical Hessian.
    
    Parameters:
    -----------
    params : ndarray
        Parameter estimates.
    log_lik : float
        Log-likelihood value.
    func : callable
        Objective function (negative log-likelihood).
    eps : float
        Perturbation size for numerical derivatives (default: 1e-5).
    
    Returns:
    --------
    dict
        Standard errors and p-values.
    """
    try:
        n = len(params)
        hessian = np.zeros((n, n))
        cache = {}  # Cache function evaluations
        
        def eval_func(x):
            x_tuple = tuple(x)
            if x_tuple not in cache:
                cache[x_tuple] = func(x)
            return cache[x_tuple]
        
        for i in range(n):
            for j in range(n):
                x_pp = params.copy()
                x_mm = params.copy()
                x_pm = params.copy()
                x_mp = params.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                f_pp = eval_func(x_pp)
                f_mm = eval_func(x_mm)
                f_pm = eval_func(x_pm)
                f_mp = eval_func(x_mp)
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
        hessian += np.eye(n) * 1e-6
        cov_matrix = np.linalg.inv(hessian)
        std_err = np.sqrt(np.abs(np.diag(cov_matrix)))
        z_scores = params / std_err
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        return {'std_err': std_err, 'p_values': p_values}
    except Exception as e:
        print(f"Error in compute_stats: {e}")
        return {'std_err': np.array([np.nan] * n), 'p_values': np.array([np.nan] * n)}

