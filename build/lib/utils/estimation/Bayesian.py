
from filters import kalman_objective
from filters import Kalman
from utils.optimizers import rwm,compute_proposal_sigma
from utils.estimation.results import compute_stats, create_results_table
import numpy as np


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
    try:
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
        obj_func = lambda params: -kalman_objective(params, fixed_params, param_names, y, update_state_space)
        
        # Run RWM
        result = rwm(obj_func, prior, x0, lb, ub, n_iter, burn_in, thin, sigma, seed, verbose)
        print("result",result)
        # Validate result
        if not isinstance(result, dict) or 'samples' not in result or 'log_posterior' not in result:
            raise ValueError(f"Invalid result from rwm: {result}")
        
        # Create table
        table = create_results_table(result,param_names,log_lik=np.mean(result['log_posterior']),obj_func=obj_func,method='RWM',prior_func=prior,output_dir='plots')
        if verbose:
            print(f"Final RWM result: {result}")
            print(f"Results table: {table}")
        
        return {'result': result, 'table': table}
    
    except Exception as e:
        error_result = {
            'samples': None,
            'log_posterior': None,
            'acceptance_rate': None,
            'message': f'RWM failed: {str(e)}'
        }
        error_table = {'Method': 'RWM', 'Message': f'Table creation failed: {str(e)}'}
        print(f"Error in rwm_kalman: {e}, returning: {error_result}, table: {error_table}")
        return {'result': error_result, 'table': error_table}