import unittest
import numpy as np
import pandas as pd
from econometron.Models.VectorAutoReg.VARMA import VARMA

class TestVARMA(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(100, 2), columns=['y1', 'y2'])
        self.data = data
        self.model = VARMA(data=data, max_p=2, max_q=2, columns=['y1', 'y2'], criterion='AIC', plot=False, check_stationarity=False)

    def test_initialization(self):
        self.assertIsInstance(self.model, VARMA)
        self.assertEqual(self.model.data.shape, (100, 2))

    def test_fit(self):
        result = self.model.fit(columns=['y1', 'y2'])
        self.assertIsNotNone(result)
        self.assertTrue(self.model.fitted)

    def test_predict(self):
        self.model.fit(columns=['y1', 'y2'])
        forecast = self.model.predict(n_periods=5, plot=False)
        self.assertIn('point', forecast)
        self.assertEqual(forecast['point'].shape[0], 5)

    def test_simulate(self):
        self.model.fit(columns=['y1', 'y2'])
        sim = self.model.simulate(n_periods=10, plot=False)

        self.assertEqual(sim.shape, (10, 2))

    def test_edge_cases(self):
        # Edge: insufficient observations
        data = pd.DataFrame(np.random.randn(2, 2), columns=['y1', 'y2'])
        with self.assertRaises(Exception):
            VARMA(data=data, max_p=2, max_q=2, columns=['y1', 'y2'], criterion='AIC', plot=False, check_stationarity=False).fit(columns=['y1', 'y2'])

if __name__ == "__main__":
    unittest.main()
