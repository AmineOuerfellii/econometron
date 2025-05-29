import numpy as np

def make_state_space_updater(
    base_params: dict,
    solver: callable,
    derived_fn: callable,
    R_builder: callable,
    C_builder: callable,
):
    """
    Creates a generalized state-space updater function.

    Parameters:
    -----------
    base_params : dict
        Default model parameters.
    solver : callable
        Function to solve the model, e.g., `Model.solve`.
    derived_fn : callable
        Function to compute derived parameters (e.g., 'Parameters that depend on other prams values').
    R_builder : callable
        Function that takes params and returns R matrix.
    C_builder : callable
        Function that takes params and returns C matrix.


    Returns:
    --------
    A function that takes a dictionary of parameter updates and returns the state-space matrices.
    """

    def update_state_space(params):
        full_params = base_params.copy()
        full_params.update(params)
        
        # Apply derived parameter logic
        derived_fn(full_params)
        
        # Solve the model
        F, P = solver(full_params)
        # D = F[::-1, :]  # Reversed F
        
        
        R = R_builder(full_params)
        RR = R @ R.T
        
        C = C_builder(full_params)
        QQ = C @ C.T
        
        return {'A': P, 'D': F, 'Q': QQ, 'R': RR}

    return update_state_space
