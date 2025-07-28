import unittest
import numpy as np
import pandas as pd
from econometron.Models.VectorAutoReg.VAR import VAR

class TestVAR(unittest.TestCase):
    def setUp(self):
        # Simple stationary data
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(100, 2), columns=['y1', 'y2'])
        self.data = data
        self.model = VAR(data, max_p=2, columns=['y1', 'y2'], criterion='AIC', plot=False, check_stationarity=False)

    def test_initialization(self):
        self.assertIsInstance(self.model, VAR)
        self.assertEqual(self.model.data.shape, (100, 2))

    def test_fit(self):
        result = self.model.fit(columns=['y1', 'y2'], output=False)
        self.assertIsNotNone(result)
        self.assertTrue(self.model.best_model is not None)

    def test_predict(self):
        self.model.fit(columns=['y1', 'y2'], output=False)
        forecast = self.model.predict(n_periods=5, plot=False)
        self.assertIn('point', forecast)
        self.assertEqual(forecast['point'].shape[0], 5)

    def test_impulse_res(self):
        self.model.fit(columns=['y1', 'y2'], output=False)
        irf = self.model.impulse_res(h=5, orth=True, bootstrap=False, plot=False)
        self.assertTrue(isinstance(irf, np.ndarray) or (isinstance(irf, tuple) and isinstance(irf[0], np.ndarray)))

    def test_fevd(self):
        self.model.fit(columns=['y1', 'y2'], output=False)
        fevd = self.model.FEVD(h=5, plot=False)
        self.assertIsInstance(fevd, np.ndarray)

    def test_simulate(self):
        self.model.fit(columns=['y1', 'y2'], output=False)
        sim = self.model.simulate(n_periods=10, plot=False)
        self.assertEqual(sim.shape, (10, 2))

    def test_edge_cases(self):
        # Edge: non-numeric columns
        data = pd.DataFrame({'y1': np.random.randn(10), 'y2': np.random.randn(10), 'cat': ['a']*10})
        with self.assertRaises(Exception):
            VAR(data, max_p=2, columns=['y1', 'cat'], criterion='AIC', plot=False, check_stationarity=False).fit(columns=['y1', 'cat'], output=False)
        # Edge: insufficient observations
        data = pd.DataFrame(np.random.randn(2, 2), columns=['y1', 'y2'])
        with self.assertRaises(Exception):
            VAR(data, max_p=2, columns=['y1', 'y2'], criterion='AIC', plot=False, check_stationarity=False).fit(columns=['y1', 'y2'], output=False)

if __name__ == "__main__":
    unittest.main()
