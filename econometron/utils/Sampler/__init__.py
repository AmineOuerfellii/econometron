from numpy.random import uniform, rand, seed
import numpy as np
from scipy.stats import norm
import numpy.random as npr
from econometron.utils.optimizers import evaluate_func

def rwm(objecti_func, prior, x0, lb, ub, n_iter=10000, burn_in=1000, thin=1, sigma=0.1, seed=42, verbose=False):
    """
    Random Walk Metropolis sampler.
    
    Parameters:
    -----------
    objecti_func : callable
        Log-likelihood function
    prior : callable
        Log-prior function (should handle bounds internally)
    x0 : array-like
        Initial parameter values
    lb, ub : array-like
        Lower and upper bounds (used only for proposal clipping)
    n_iter : int
        Total number of iterations
    burn_in : int
        Number of burn-in iterations
    thin : int
        Thinning interval
    sigma : float or array-like
        Proposal step size(s)
    seed : int
        Random seed
    verbose : bool
        Print debug information
    
    Returns:
    --------
    dict : Results dictionary
    """
    try:
        np.random.seed(seed)
        x = np.array(x0, dtype=float)
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)
        sigma = np.array([sigma] * len(x) if np.isscalar(sigma) else sigma, dtype=float)
        N = len(x)

        # Input validation
        if len(lb) != N or len(ub) != N or len(sigma) != N:
            return {
                'samples': None, 'log_posterior': None, 'acceptance_rate': 0,
                'message': "Invalid input dimensions."
            }

        # Initialize arrays
        samples = np.zeros((n_iter // thin, N))
        log_posterior = np.zeros(n_iter // thin)
        n_accept = 0
        x_current = x.copy()
        
        # Evaluate initial log-likelihood and prior
        log_like_current = evaluate_func(objecti_func, x_current)
        log_prior_current = prior(x_current)
        if not (np.isfinite(log_like_current) and np.isfinite(log_prior_current)):
                # Try to find a valid starting point by sampling from bounds
            max_attempts = 100
            for attempt in range(max_attempts):
                x_current = np.random.uniform(lb, ub)
                log_like_current = evaluate_func(objecti_func, x_current)
                log_prior_current = prior(x_current)
                if np.isfinite(log_like_current) and np.isfinite(log_prior_current):
                    if verbose:
                        print(f"Found valid starting point after {attempt + 1} attempts: {x_current}")
                    break
            else:
                return {
                    'samples': None, 'log_posterior': None, 'acceptance_rate': 0,
                    'message': "Could not find valid initial point after 100 attempts."
                }
         # Evaluate initial log-posterior
        log_post_current = log_like_current + log_prior_current
        # Main MCMC loop
        sample_idx = 0
        for i in range(n_iter):
            # Propose new state (clip to bounds to avoid immediate rejection)
            x_proposed = np.clip(x_current + np.random.normal(0, sigma, N), lb, ub)
            
            # Evaluate proposed log-posterior
            log_like_proposed = evaluate_func(objecti_func, x_proposed)
            log_prior_proposed = prior(x_proposed)
            
            # Check if proposal is valid
            if np.isfinite(log_like_proposed) and np.isfinite(log_prior_proposed):
                log_post_proposed = log_like_proposed + log_prior_proposed
                # Compute acceptance ratio
                log_ratio = log_post_proposed - log_post_current
                accept_prob = np.exp(min(0, log_ratio))
            else:
                accept_prob = 0
                log_post_proposed = -np.inf
            
            # Accept/reject step
            if np.random.rand() <= accept_prob:
                x_current = x_proposed
                log_post_current = log_post_proposed
                n_accept += 1
                if verbose and i % 1000 == 0:
                    print(f'Iteration {i}: Accepted, log_post = {log_post_current:.3f}')

            # Adaptive step size during burn-in
            if i < burn_in and i % 50 == 0 and i > 0:
                accept_rate = n_accept / (i + 1)
                if accept_rate < 0.2:
                    sigma *= 0.8  # Decrease step size
                elif accept_rate > 0.5:
                    sigma *= 1.2  # Increase step size
                if verbose:
                    print(f"Iteration {i}: Accept rate = {accept_rate:.3f}, Updated sigma = {sigma}")
            
            # Store samples after burn-in
            if i >= burn_in and i % thin == 0:
                if sample_idx < len(samples):
                    samples[sample_idx] = x_current
                    log_posterior[sample_idx] = log_post_current
                    sample_idx += 1

        acceptance_rate = n_accept / n_iter

        res = {
            'samples': samples,
            'log_posterior': log_posterior,
            'acceptance_rate': acceptance_rate,
            'mean_posterior_parameters': np.mean(samples, axis=0),
            'std_posterior_parameters': np.std(samples, axis=0),
            'message': 'RWM completed successfully.'
        }

        if verbose:
            print("RWM Summary:")
            print(f"Acceptance rate: {acceptance_rate:.3f}")
            print(f"Mean posterior parameters: {res['mean_posterior_parameters']}")
            print(f"Std posterior parameters: {res['std_posterior_parameters']}")

        return res

    except Exception as e:
        print(f"Error in rwm: {e}")
        return None