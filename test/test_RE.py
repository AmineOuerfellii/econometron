import unittest
import numpy as np
import pandas as pd
import econometron
from econometron.Models import RE_model as Model
from econometron.utils.estimation.Bayesian import rwm_kalman
from econometron.utils.estimation.MLE import genetic_algorithm_kalman, simulated_annealing_kalman
from econometron.utils.state_space.update_ss import make_state_space_updater
from econometron.utils.estimation.prior import make_prior_function
from econometron.utils.estimation.Bayesian import compute_proposal_sigma

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(
            equations=[
            "x_t - x_tp1 + (1/g) * (r_t - p_tp1) = 0",
            "p_t - beta * p_tp1 - kappa * (x_t - xbar_t) = 0",
            "- r_t + phi*p_t=0",
            "- xbar_tp1 + rho * xbar_t + sigmax = 0"],
            variables = ['x', 'p', 'r','xbar'],
            exo_states=['xbar'],
            shocks= ['sigmax'],
            parameters = {
            'g': 5,      # Inverse of relative risk aversion (1/g)
            'beta': 0.99,       # Discount factor
            'kappa': 0.88,
            'rho': 0.95,        # Persistence of output gap target
            'phi': 1.5,         # Taylor rule inflation coefficient
            'd': 0.5,          # Calvo parameter
            'sigmax':0.01
            }
        )

    def test_compute_ss(self):
        try:
            self.model.compute_ss()
        except Exception as e:
            self.fail(f"compute_ss raised an exception: {e}")

    def test_approximate(self):
        try:
            self.model.compute_ss()
            self.model.approximate()
        except Exception as e:
            self.fail(f"approximate raised an exception: {e}")
    
    def test_simulate(self):
        self.model.compute_ss()
        self.model.approximate()
        self.model.solve_RE_model()
        self.model.simulate(T=10)
        self.assertTrue(hasattr(self.model, "simulated"))
        self.assertIsInstance(self.model.simulated, pd.DataFrame)

    def test_irfs(self):
        self.model.compute_ss()
        self.model.approximate()
        self.model.solve_RE_model()
        self.model._compute_irfs(T=10)
        self.assertTrue(hasattr(self.model, "irfs"))
        self.assertIsInstance(self.model.irfs, dict)

    def test_plot_irfs(self):
        self.model.compute_ss()
        self.model.approximate()
        self.model.solve_RE_model()
        self.model._compute_irfs(T=10)
        try:
            self.model.plot_irfs()
        except Exception as e:
            self.fail(f"plot_irfs raised an exception: {e}")

    def test_genetic_algorithm_kalman_runs(self):
        base_params = {'a': 1.0, 'b': -1.0}
        def solver(params):
            return np.eye(2), np.eye(2)
        def build_R(params):
            return np.eye(2)
        def build_C(params):
            return np.eye(2)
        update_state_space = make_state_space_updater(
            base_params=base_params,
            solver=solver,
            build_R=build_R,
            build_C=build_C
        )
        y = np.zeros((2, 10))
        x0 = [1.0, -1.0]
        lb = [-5, -5]
        ub = [5, 5]
        param_names = ['a', 'b']
        fixed_params = {}
        result = genetic_algorithm_kalman(y, x0, lb, ub, param_names, fixed_params, update_state_space, pop_size=10, n_gen=5, verbose=False)
        self.assertIn('result', result)
        self.assertIn('x', result['result'])

    def test_simulated_annealing_kalman_runs(self):
        base_params = {'a': 1.0, 'b': -1.0}
        def solver(params):
            return np.eye(2), np.eye(2)
        def build_R(params):
            return np.eye(2)
        def build_C(params):
            return np.eye(2)
        update_state_space = make_state_space_updater(
            base_params=base_params,
            solver=solver,
            build_R=build_R,
            build_C=build_C
        )
        y = np.zeros((2, 10))
        x0 = [1.0, -1.0]
        lb = [-5, -5]
        ub = [5, 5]
        param_names = ['a', 'b']
        fixed_params = {}
        result = simulated_annealing_kalman(y, x0, lb, ub, param_names, fixed_params, update_state_space, T0=5, rt=0.9, nt=2, ns=2, seed=42, max_evals=100, eps=0.01)
        self.assertIn('result', result)
        self.assertIn('x', result['result'])

    def test_rwm_kalman_runs(self):
        base_params = {'a': 1.0, 'b': -1.0}
        def solver(params):
            return np.eye(2), np.eye(2)
        def build_R(params):
            return np.eye(2)
        def build_C(params):
            return np.eye(2)
        update_state_space = make_state_space_updater(
            base_params=base_params,
            solver=solver,
            build_R=build_R,
            build_C=build_C
        )
        y = np.zeros((2, 10))
        x0 = [1.0, -1.0]
        lb = [-5, -5]
        ub = [5, 5]
        param_names = ['a', 'b']
        fixed_params = {}
        # Use uniform prior and computed sigma
        bounds = {'a': (-5, 5), 'b': (-5, 5)}
        priors = {'a': (lambda x: 0, {}), 'b': (lambda x: 0, {})}
        prior = make_prior_function(param_names, priors, bounds)
        sigma = compute_proposal_sigma(2, np.array(lb), np.array(ub), [0.1, 0.1])
        result = rwm_kalman(y, x0, lb, ub, param_names, fixed_params, update_state_space, n_iter=20, burn_in=5, thin=2, sigma=sigma, prior=prior, verbose=False)
        self.assertIn('result', result)
        self.assertIn('samples', result['result'])

if __name__ == '__main__':
    unittest.main()
