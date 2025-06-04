from filters import kalman_objective
from filters import Kalman
from utils.optimizers import genetic_algorithm, simulated_annealing
from utils.estimation import create_results_table
import numpy as np


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
    try:
        obj_func = lambda params: kalman_objective(params, fixed_params, param_names, y, update_state_space)
        result = genetic_algorithm(obj_func, x0, lb, ub, pop_size, n_gen, crossover_rate, 
                                  mutation_rate, elite_frac, seed, verbose)
        if verbose:
            print(f"GA result: {result}")
        table = create_results_table(result, param_names, -result['fun'] if result['fun'] is not None else np.nan, 
                                    obj_func, 'Genetic Algorithm')
        if verbose:
            print(f"Results table: {table}")
        return {'result': result, 'table': table}
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
    obj_func = lambda params:kalman_objective(params, fixed_params, param_names, y, update_state_space)
    result = simulated_annealing(obj_func, x0, lb, ub, T0, rt, nt, ns, seed,max_evals,eps)
    table = create_results_table(result, param_names, -result['fun'] if result['fun'] is not None else np.nan, obj_func, 'Simulated Annealing')
    return {'result': result , 'table': table}