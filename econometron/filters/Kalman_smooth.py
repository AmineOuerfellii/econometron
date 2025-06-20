
from . import Kalman

def kalman_smooth(y, update_state_space):
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
    print("Running Kalman smoother...")
    # Update state-space matrices
    ss_params = update_state_space
    # Run Kalman filter
    try:
        kalman = Kalman(ss_params)
        result = kalman.smooth(y)
        smooth_state = result['Xsm']
        return smooth_state
    except Exception as e:
        print("Error in kalman_smooth:")
        print(f"Exception: {e}")
        return None