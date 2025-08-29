from econometron.filters import kalman_objective
from econometron.filters import Kalman
from econometron.utils.optimizers import genetic_algorithm, simulated_annealing
import numpy as np
import pandas as pd

####################################### Genetic Algorithm #######################################
# Genetic Algorithm for Maximum Likelihood Estimation
def genetic_algorithm_kalman(
    y,
    x0,
    lb,
    ub,
    param_names,
    fixed_params,
    update_state_space,
    pop_size=50,
    n_gen=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elite_frac=0.1,
    seed=24,
    verbose=True
):
    """
    Genetic Algorithm for DSGE model parameter estimation using Kalman filter.
    
    Returns:
    --------
    dict
        Dictionary with optimized parameters, objective value, nfev, and message
    """
    if isinstance(y, np.ndarray):
        if y.shape[0] > y.shape[1]:
            y = y.T
    elif isinstance(y, pd.DataFrame):
        if y.shape[0] > y.shape[1]:
            y = y.T.values
        else:
            y=y.values
    try:
        obj_func = lambda params: kalman_objective(params, fixed_params, param_names, y, update_state_space)
        result = genetic_algorithm(obj_func, x0, lb, ub, pop_size, n_gen, crossover_rate, 
                                  mutation_rate, elite_frac, seed, verbose)
        if verbose:
            print(f"GA result: {result}")
        return result
    except Exception as e:
        error_result = {
            'x': None,
            'fun': None,
            'nfev': None,
            'message': f'GA Kalman failed: {str(e)}'
        }
        print(f"Error in genetic_algorithm_kalman: {e}, returning: {error_result}")
        return error_result
################################################SiM ANN #######################################################
# Wrapper for Simulated Annealing with Kalman filter
def simulated_annealing_kalman(
    y,
    x0,
    lb,
    ub,
    param_names,
    fixed_params,
    update_state_space,
    T0=5,
    rt=0.9,
    nt=5,
    ns=10,
    seed=42,
    max_evals=1000000,
    eps=0.001
):
    if isinstance(y, np.ndarray):
        if y.shape[0] > y.shape[1]:
            y = y.T
    elif isinstance(y, pd.DataFrame):
        if y.shape[0] > y.shape[1]:
            y = y.T.values
        else:
            y=y.values
    print(f"Starting Simulated Annealing with params: T0: {T0}, rt: {rt}, nt: {nt}, ns: {ns}, seed: {seed}")
    obj_func = lambda params:kalman_objective(params, fixed_params, param_names, y, update_state_space)
    result = simulated_annealing(obj_func, x0, lb, ub, T0, rt, nt, ns, seed,max_evals,eps)
    return result