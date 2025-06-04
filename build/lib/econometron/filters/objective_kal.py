from . import Kalman 
import numpy as np

def kalman_objective(params, fixed_params, param_names, y, update_state_space):
    """
    Objective function for Kalman filter optimization.

    Parameters:
    -----------
    params : ndarray
        Parameters to optimize.
    fixed_params : dict
        Fixed parameters and their values.
    param_names : list
        Names of parameters to optimize.
    y : ndarray
        Observations (m x T).
    update_state_space : callable
        Function to update state-space matrices given parameters.

    Returns:
    --------
    float
        Negative log-likelihood.
    """
    # Combine optimized and fixed parameters
    full_params = fixed_params.copy()
    for name, value in zip(param_names, params):
        full_params[name] = value

    # Run Kalman filter
    try:
        ss_params = update_state_space(full_params)
        kalman = Kalman(ss_params)
        result = kalman.filter(y)
        log_lik = result['log_lik']
        return log_lik
    except Exception as e:
        print("Error in kalman_objective:")
        print(f"Params: {params}")
        print(f"Exception: {e}")
        return 8e30