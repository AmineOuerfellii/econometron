import unittest
import numpy as np
import pandas as pd
import econometrica as ec
from ec import Model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(
            equations=[
            "x_t - x_tp1 + (1/g) * (r_t - p_tp1) = 0",
            "p_t - beta * p_tp1 - kappa * (x_t - xbar_t) = 0",
            "- r_t + phi*p_t=0",
            "- xbar_tp1 + rho * xbar_t + sigmax = 0"],
            variables = ['x', 'p', 'r','xbar'],
            states = ['xbar'],
            exo_states=['xbar'],
            shock_names = ['sigmax'],
            parameters = {
            'g': 5,      # Inverse of relative risk aversion (1/g)
            'beta': 0.99,       # Discount factor
            'kappa': 0.88,
            'rho': 0.95,        # Persistence of output gap target
            'phi': 1.5,         # Taylor rule inflation coefficient
            'd': 0.5,          # Calvo parameter
            'sigmax':0.01
            },
            log_linear=False,
            shock_variance={'eps': 0.01},
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
        self.model.simulate(T=10)
        self.assertTrue(hasattr(self.model, "simulated"))
        self.assertIsInstance(self.model.simulated, pd.DataFrame)

    def test_irfs(self):
        self.model.compute_ss()
        self.model.approximate()
        self.model.compute_irfs(T=10)
        self.assertTrue(hasattr(self.model, "irfs"))
        self.assertIsInstance(self.model.irfs, dict)

    def test_plot_irfs(self):
        self.model.compute_ss()
        self.model.approximate()
        self.model.compute_irfs(T=10)
        try:
            self.model.plot_irfs()
        except Exception as e:
            self.fail(f"plot_irfs raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
