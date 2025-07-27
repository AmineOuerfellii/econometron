import unittest
import numpy as np
import pandas as pd
from econometron.Models.dynamicsge import linear_dsge

class TestLinearDSGE(unittest.TestCase):
    def setUp(self):
        # Minimal model setup
        self.equations = ["y_t = a * y_tm1 + e_t"]
        self.variables = ["y"]
        self.exo_states = []
        self.endo_states = ["y"]
        self.parameters = {"a": 0.9}
        self.shocks = ["e_t"]
        self.model = linear_dsge(
            equations=self.equations,
            variables=self.variables,
            exo_states=self.exo_states,
            endo_states=self.endo_states,
            parameters=self.parameters,
            shocks=self.shocks
        )

    def test_initialization(self):
        self.assertEqual(self.model.n_vars, 1)
        self.assertEqual(self.model.n_equations, 1)
        self.assertEqual(self.model.n_states, 1)
        self.assertEqual(self.model.n_controls, 0)

    def test_set_initial_guess(self):
        self.model.set_initial_guess([0.0])
        self.assertTrue(hasattr(self.model, 'initial_guess'))
        with self.assertRaises(ValueError):
            self.model.set_initial_guess([0.0, 1.0])  # Wrong length
        with self.assertRaises(ValueError):
            self.model.set_initial_guess("not_a_list")  # Not a list

    def test_parse_equations(self):
        tp1, t, shock = self.model._parse_equations(self.equations[0])
        self.assertIsNotNone(tp1)
        self.assertIsNotNone(t)
        self.assertIsNotNone(shock)

    def test_equations(self):
        res = self.model.equations([1.0], [1.0], {"a": 0.9, "e_t": 0.0})
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1,))

    def test_compute_ss(self):
        ss = self.model.compute_ss(guess=[0.0])
        self.assertIsInstance(ss, pd.Series)
        self.assertEqual(ss.shape[0], 1)

    def test_analytical_jacobians_error(self):
        # Should raise if steady state not computed
        model = linear_dsge(
            equations=self.equations,
            variables=self.variables,
            exo_states=self.exo_states,
            endo_states=self.endo_states,
            parameters=self.parameters,
            shocks=self.shocks
        )
        with self.assertRaises(ValueError):
            model._Analytical_jacobians()

    def test_approximate_and_solve(self):
        self.model.compute_ss(guess=[0.0])
        A, B, C = self.model.approximate(method='analytical')
        self.assertEqual(A.shape, (1, 1))
        self.assertEqual(B.shape, (1, 1))
        self.assertEqual(C.shape, (1, 1))
        F, P = self.model.solve_RE_model()
        self.assertIsNotNone(F)
        self.assertIsNotNone(P)

    def test_irfs_and_plot(self):
        self.model.compute_ss(guess=[0.0])
        self.model.approximate(method='analytical')
        self.model.solve_RE_model()
        irfs = self.model._compute_irfs(T=5, t0=1)
        self.assertIsInstance(irfs, dict)
        for shock, df in irfs.items():
            self.assertIsInstance(df, pd.DataFrame)

    def test_edge_cases(self):
        # Edge: No equations
        with self.assertRaises(Exception):
            linear_dsge(equations=[], variables=["y"], exo_states=[], endo_states=["y"], parameters={"a": 0.9}, shocks=["e_t"])
        # Edge: No variables
        with self.assertRaises(Exception):
            linear_dsge(equations=["y_t = a * y_tm1 + e_t"], variables=[], exo_states=[], endo_states=[], parameters={"a": 0.9}, shocks=["e_t"])

if __name__ == "__main__":
    unittest.main()
