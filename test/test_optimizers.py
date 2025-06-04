import numpy as np
import pytest
from utils.optimizers.optim import simulated_annealing
from utils.optimizers.eval import evaluate_func
from utils.estimation.Bayesian import rwm_kalman
from utils.estimation.MLE import genetic_algorithm_kalman

def test_simulated_annealing_basic():
    # Simple quadratic function: minimum at x=0
    func = lambda x: np.sum(np.square(x))
    x0 = [5.0, -3.0]
    lb = [-10, -10]
    ub = [10, 10]
    T = 10
    cooling_rate = 0.9
    n_temp = 5
    n_steps = 10
    seed = 42
    max_evals = 1000
    result = simulated_annealing(func, x0, lb, ub, T, cooling_rate, n_temp, n_steps, seed, max_evals)
    assert 'x' in result
    assert np.allclose(result['x'], [0, 0], atol=1)
    assert result['fun'] < 1.0

def test_evaluate_func():
    func = lambda x: sum(x)
    params = [1, 2, 3]
    val = evaluate_func(func, params)
    assert val == 6
    # Non-callable returns inf
    val2 = evaluate_func(None, params)
    assert val2 == float('inf')

def test_genetic_algorithm_kalman_runs():
    # Dummy Kalman objective: minimum at [0,0]
    def dummy_update_state_space(params):
        return {'A': np.eye(2), 'D': np.eye(2), 'Q': np.eye(2), 'R': np.eye(2)}
    y = np.zeros((2, 10))
    x0 = [1.0, -1.0]
    lb = [-5, -5]
    ub = [5, 5]
    param_names = ['a', 'b']
    fixed_params = {}
    result = genetic_algorithm_kalman(y, x0, lb, ub, param_names, fixed_params, dummy_update_state_space, pop_size=10, n_gen=5, verbose=False)
    assert 'result' in result
    assert 'x' in result['result']

def test_rwm_kalman_runs():
    # Dummy Kalman objective: minimum at [0,0]
    def dummy_update_state_space(params):
        return {'A': np.eye(2), 'D': np.eye(2), 'Q': np.eye(2), 'R': np.eye(2)}
    y = np.zeros((2, 10))
    x0 = [1.0, -1.0]
    lb = [-5, -5]
    ub = [5, 5]
    param_names = ['a', 'b']
    fixed_params = {}
    result = rwm_kalman(y, x0, lb, ub, param_names, fixed_params, dummy_update_state_space, n_iter=50, burn_in=10, thin=2, verbose=False)
    assert 'result' in result
    assert 'samples' in result['result']
