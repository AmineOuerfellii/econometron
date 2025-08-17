
import numpy as np
from econometron.filters import kalman_objective
import econometron.utils.Sampler as sampler
from econometron.utils.estimation.results import create_results_table
from typing import List, Tuple, Dict, Callable
from scipy.stats import gamma, beta as beta_dist

### Compute proposal sigma
def compute_proposal_sigma(n_params, lb, ub, base_std=0.1):
    """
    Compute proposal standard deviations based on parameter bounds.

    Parameters:
    -----------
    n_params : int
        Number of parameters.
    lb : ndarray
        Lower bounds for parameters.
    ub : ndarray
        Upper bounds for parameters.
    base_std : float or ndarray, optional
        Base standard deviation (scalar or array of length n_params, default: 0.1).

    Returns:
    --------
    ndarray
        Proposal standard deviations (shape: n_params).
    """
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    if len(lb) != n_params or len(ub) != n_params:
        raise ValueError("lb and ub must have length n_params")
    ranges = ub - lb
    ranges = np.where(ranges > 0, ranges, 1.0)  # Avoid zero ranges
    if np.isscalar(base_std):
        sigma = base_std * ranges / 10  # Scale to 10% of range
    else:
        base_std = np.array(base_std, dtype=float)
        if len(base_std) != n_params:
            raise ValueError("base_std must have length n_params")
        sigma = base_std * ranges / 10
    return sigma
#######################################Make Prior func #############################################################

def make_prior_function(
    param_names: List[str],
    priors: Dict[str, Tuple[Callable, Dict]],
    bounds: Dict[str, Tuple[float, float]],
    verbose: bool = False
):
    """
    Create a generalized log-prior function for a model.
    
    Parameters:
    -----------
    param_names : list of str
        Names of the parameters in the order they appear in the vector.
    priors : dict
        Mapping from parameter name to a tuple (distribution, parameters),
        e.g., 'beta': (beta_dist, {'a': 99, 'b': 1})
    bounds : dict
        Mapping from parameter name to (lower_bound, upper_bound)
    verbose : bool
        Whether to print debug output.
    
    Returns:
    --------
    Function that takes a parameter vector and returns the log-prior.
    """
    
    def prior(params: List[float]) -> float:
        if len(params) != len(param_names):
            if verbose:
                print("Error: Parameter vector length mismatch.")
            return -np.inf
        
        # Check bounds and compute log-prior in single pass
        log_priors = []
        for name, value in zip(param_names, params):
            lb, ub = bounds[name]
            # Return -inf if outside bounds
            if not (lb < value < ub):
                if verbose:
                    print(f"[Bound Error] {name} = {value:.4f} not in ({lb}, {ub})")
                return -np.inf
            
            # Compute log-prior for this parameter
            dist, kwargs = priors[name]
            try:
                logp = dist.logpdf(value, **kwargs)
                if not np.isfinite(logp):
                    if verbose:
                        print(f"[PDF Error] {name}: Non-finite logpdf value")
                    return -np.inf
                log_priors.append(logp)
                if verbose:
                    print(f"[Log Prior] {name}: logpdf({value:.4f}) = {logp:.3f}")
            except Exception as e:
                if verbose:
                    print(f"[PDF Error] {name}: {e}")
                return -np.inf
        
        total_log_prior = sum(log_priors)
        if verbose:
            print(f"[Total Log Prior] = {total_log_prior:.3f} | Params = {params}")
        
        return total_log_prior if np.isfinite(total_log_prior) else -np.inf
    
    return prior

####################################### Random Walk Metropolis (RWM) #######################################
  #deafault values for the parameters
  # for the random walk metropolis
def rwm_kalman(
    y,
    x0,
    lb,
    ub,
    param_names,
    fixed_params,
    update_state_space,
    n_iter=10000,
    burn_in=1000,
    thin=1,
    sigma=None,
    base_std=0.1,            
    seed=42,
    verbose=True,
    prior=None
):
    """
    Random Walk Metropolis for DSGE model estimation using Kalman filter.

    Parameters:
    -----------
    y : ndarray
        Observations (m x T).
    x0 : ndarray
        Initial parameter vector.
    lb : ndarray
        Lower bounds for parameters.
    ub : ndarray
        Upper bounds for parameters.
    param_names : list
        Names of parameters to estimate.
    fixed_params : dict
        Fixed parameters for the DSGE model.
    update_state_space : callable
        Function to update state-space matrices.
    n_iter : int, optional
        Number of MCMC iterations (default: 10000).
    burn_in : int, optional
        Number of burn-in iterations (default: 1000).
    thin : int, optional
        Thinning factor (default: 1).
    sigma : float or ndarray, optional
        Proposal standard deviation (scalar or per-parameter). If None, computed based on bounds.
    base_std : float or ndarray, optional
        Base standard deviation for computing sigma (default: 0.1).
    seed : int, optional
        Random seed (default: 42).
    verbose : bool, optional
        Print summary statistics if True (default: True).
    prior : callable, optional
        Prior function returning log-prior probability (default: defined above).

    Returns:
    --------
    dict
        - result: Dictionary with samples, log_posterior, acceptance_rate, message.
        - table: Dictionary with Parameter, Estimate, Std Error, Log-Likelihood, Method.
    """
    #try:
    # Validate inputs
    x0 = np.array(x0, dtype=float)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    N = len(x0)
    if len(lb) != N or len(ub) != N or len(param_names) != N:
        raise ValueError("Length mismatch in x0, lb, ub, or param_names")
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError(f"Initial parameters outside bounds: x0={x0}, lb={lb}, ub={ub}")
    
    # Compute proposal sigma
    if sigma is None:
        b_std=[base_std]*N
        sigma = compute_proposal_sigma(N, lb, ub, b_std)
    sigma = np.array(sigma, dtype=float)
    if sigma.size == 1:
        sigma = np.full(N, sigma)
    if sigma.size != N:
        raise ValueError("Sigma length does not match parameter vector length")
    if prior is None:
        prior = lambda params: 0 if np.all((params >= lb) & (params <= ub)) else -np.inf  # Uniform prior
    else:
        prior=prior
    # Define objective function
    obj_func = lambda params: - kalman_objective(params, fixed_params, param_names, y, update_state_space)
    
    # Run RWM
    result = sampler(obj_func, prior, x0, lb, ub, n_iter, burn_in, thin, sigma, seed, verbose)
    print(result)
    # Validate result
    if not isinstance(result, dict) or 'samples' not in result or 'log_posterior' not in result:
        raise ValueError(f"Invalid result from rwm: {result}")
    # Create table
    table = create_results_table(result, param_names, log_lik=np.mean(result['log_posterior']), obj_func=obj_func, method='RWM', prior_func=prior)
    print('table', table)
    if verbose:
        print(f"Final RWM result: {result}")
        print(f"Results table: {table}")

    return {'result': result, 'table': table}
