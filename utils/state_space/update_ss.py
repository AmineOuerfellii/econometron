import numpy as np

def make_state_space_updater(
    base_params: dict,
    solver: callable,
    build_R: callable,
    build_C: callable,
    derived_fn: callable = None,
):
    """
    Creates a generalized state-space updater function.

    Parameters:
    -----------
    base_params : dict
        Default model parameters.
    solver : callable
        Function to solve the model, e.g., `Model.solve`.
    derived_fn : callable, optional
        Function to compute derived parameters (e.g., 'Parameters that depend on other prams values').
    build_R : callable
        Function that takes params and returns R matrix.
    build_C : callable
        Function that takes params and returns C matrix.

    Returns:
    --------
    A function that takes a dictionary of parameter updates and returns the state-space matrices.
    """

    def update_state_space(params):
        full_params = base_params.copy()
        full_params.update(params)
        
        # Apply derived parameter logic if provided
        if derived_fn is not None:
            derived_fn(full_params)
        
        # Solve the model
        F, P = solver(full_params)
        R = build_R(full_params)
        RR = R @ R.T
        
        C = build_C(full_params)
        QQ = C @ C.T
        
        return {'A': P, 'D': F, 'Q': QQ, 'R': RR}

    return update_state_space