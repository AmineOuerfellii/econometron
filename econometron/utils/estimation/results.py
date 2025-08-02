# Statistics and Results Table
from scipy.stats import norm
import os
import contextlib
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

def create_results_table(
    result,
    param_names,
    log_lik,
    obj_func,
    method,
    prior_func=None,
    true_posterior_params=None,
    samples=None,
):
    """
    Create a results table for optimization or sampling methods, including posterior plots for Bayesian methods.

    Parameters:
    -----------
    result : dict
        Result dictionary from optimization ('x', 'fun') or sampling ('samples', 'log_posterior').
    param_names : list
        Names of parameters.
    log_lik : float
        Log-likelihood value (for optimization) or mean log-posterior (for sampling).
    obj_func : callable
        Objective function (typically negative log-likelihood).
    method : str
        Method name ('RWM', 'Genetic Algorithm', 'Simulated Annealing').
    prior_func : callable, optional
        Function to compute log-prior, signature: prior_func(params).
    true_posterior_params : dict, optional
        Parameters for true posterior (e.g., {'mean': [...], 'std': [...]} for normal).
    samples : ndarray, optional
        Posterior samples for sampling methods (if not in result['samples']).

    Returns:
    --------
    pd.DataFrame
        Table with Parameter, Estimate, Std Error/Credible Intervals, P-Value, Log-Prior,
        Log-Likelihood, Method, and embedded posterior plots for Bayesian methods.
    """
    # Initialize table components
    n_params = len(param_names)
    estimates = [np.nan] * n_params
    std_err = [np.nan] * n_params
    p_values = [np.nan] * n_params
    credible_lower = [np.nan] * n_params
    credible_upper = [np.nan] * n_params
    log_prior = np.nan
    log_like = log_lik
    plot_data = [None] * n_params

    silent_stdout = contextlib.redirect_stdout(io.StringIO())
    with silent_stdout:
        if method == 'RWM' and (result.get('samples') is not None or samples is not None):
            # Sampling methods (RWM)
            samples = result.get('samples') if samples is None else samples
            mean_estimates = np.mean(samples, axis=0)
            credible_intervals = np.percentile(samples, [2.5, 97.5], axis=0)
            log_like = np.max(result.get('log_posterior', log_like))
            estimates = mean_estimates
            credible_lower = credible_intervals[0]
            credible_upper = credible_intervals[1]

            if prior_func is not None:
                log_prior = prior_func(mean_estimates)

            # Generate posterior plots
            for i, param in enumerate(param_names):
                fig = plt.figure(figsize=(8, 6))
                gs = GridSpec(2, 1, height_ratios=[3, 1])
                
                # Posterior plot
                ax1 = fig.add_subplot(gs[0])
                counts, bins, _ = ax1.hist(samples[:, i], bins=30, density=True, alpha=0.6, color='skyblue', label='Posterior')
                ax1.set_ylabel('Density', fontsize=12)
                ax1.set_title(f'Posterior Analysis for {param}', fontsize=14, pad=15)
                
                if true_posterior_params is not None:
                    true_mean = true_posterior_params.get('mean', [0.0] * n_params)[i]
                    true_std = true_posterior_params.get('std', [1.0] * n_params)[i]
                    x = np.linspace(true_mean - 4 * true_std, true_mean + 4 * true_std, 100)
                    ax1.plot(x, norm.pdf(x, true_mean, true_std), 'r-', lw=2, label='True Posterior')

                if prior_func is not None:
                    x_min = np.min(samples[:, i]) * 0.9
                    x_max = np.max(samples[:, i]) * 1.1
                    x = np.linspace(x_min, x_max, 100)
                    prior_density = np.zeros_like(x)
                    for j, x_val in enumerate(x):
                        params = mean_estimates.copy()
                        params[i] = x_val
                        log_p = prior_func(params)
                        prior_density[j] = np.exp(log_p) if np.isfinite(log_p) else 0
                    if np.sum(prior_density) > 0:
                        prior_density /= np.trapz(prior_density, x)
                    ax1.plot(x, prior_density, 'b--', lw=2, label='Prior')

                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                # Box plot for credible intervals
                ax2 = fig.add_subplot(gs[1])
                ax2.boxplot(samples[:, i], vert=False, widths=0.4, patch_artist=True,
                           boxprops=dict(facecolor='lightgreen', alpha=0.5))
                ax2.set_yticks([])
                ax2.set_xlabel(param, fontsize=12)
                
                plt.tight_layout()
                # Convert plot to a format suitable for DataFrame (e.g., base64 or matplotlib figure object)
                plot_data[i] = fig
                plt.close(fig)

        elif result.get('x') is not None:
            # Optimization methods
            stats = compute_stats(result['x'], log_lik, obj_func)
            std_err = stats['std_err']
            p_values = stats['p_values']
            estimates = result['x']
            if prior_func is not None:
                log_prior = prior_func(result['x'])
            log_like = -obj_func(result['x']) if log_like is None else log_like

    # Create DataFrame
    df = pd.DataFrame({
        'Parameter': param_names,
        'Estimate': estimates,
        'Std Error': std_err if method in ['Genetic Algorithm', 'Simulated Annealing'] else [np.nan] * n_params,
        'P-Value': p_values if method in ['Genetic Algorithm', 'Simulated Annealing'] else [np.nan] * n_params,
        '95% Credible Interval Lower': credible_lower if method == 'RWM' else [np.nan] * n_params,
        '95% Credible Interval Upper': credible_upper if method == 'RWM' else [np.nan] * n_params,
        'Log-Prior': [log_prior] * n_params,
        'Log-Likelihood': [log_like] * n_params,
        'Posterior Plot': plot_data if method == 'RWM' else [None] * n_params,
        'Method': [method] * n_params
    })

    # Remove all-NaN columns
    df = df.dropna(how='all', axis=1)

    return df