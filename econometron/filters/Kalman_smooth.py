
from . import Kalman

def kal_smooth(params, fixed_params, param_names, y, update_state_space):
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
        smoothed state.
    """
    # Combine optimized and fixed parameters
    full_params = fixed_params.copy()
    for name, value in zip(param_names, params):
        full_params[name] = value

    # Update state-space matrices
    ss_params = update_state_space(full_params)
    # Run Kalman filter
    try:
        kalman = Kalman(ss_params)
        result = kalman.smooth(y)
        smooth_state = result['Xsm']
        return smooth_state
    except Exception as e:
        print("Error in kalman_smooth:")
        print(f"Params: {params}")
        print(f"Exception: {e}")
        return None