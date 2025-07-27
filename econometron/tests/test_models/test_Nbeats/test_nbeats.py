import unittest
import numpy as np
import torch
from econometron.Models.Neuralnets.n_beats import N_beats, NeuralForecast, LossCalculator

class TestNBeats(unittest.TestCase):
    def setUp(self):
        self.stack_configs = [
            {'n_blocks': 2, 'basis_type': 'generic', 'n_layers_per_block': 2, 'hidden_size': 8, 'share_weights': True}
        ]
        self.backcast_length = 5
        self.forecast_length = 2
        self.model = N_beats(self.stack_configs, self.backcast_length, self.forecast_length)
        self.x = torch.randn(4, self.backcast_length)

    def test_forward_shape(self):
        forecast = self.model(self.x)
        self.assertEqual(forecast.shape, (4, self.forecast_length))

    def test_model_info(self):
        info = self.model.get_model_info()
        self.assertIn('backcast_length', info)
        self.assertIn('forecast_length', info)
        self.assertIn('num_stacks', info)

    def test_loss_calculator(self):
        y_true = torch.randn(10, 2)
        y_pred = torch.randn(10, 2)
        self.assertIsInstance(LossCalculator.mse_loss(y_true, y_pred).item(), float)
        self.assertIsInstance(LossCalculator.mae_loss(y_true, y_pred).item(), float)
        self.assertIsInstance(LossCalculator.mape_loss(y_true, y_pred).item(), float)
        self.assertIsInstance(LossCalculator.smape_loss(y_true, y_pred).item(), float)
        self.assertIsInstance(LossCalculator.huber_loss(y_true, y_pred).item(), float)

    def test_neuralforecast_init(self):
        nf = NeuralForecast(self.stack_configs, self.backcast_length, self.forecast_length, device='cpu')
        self.assertIsInstance(nf.model, N_beats)

    def test_edge_cases(self):
        # Edge: zero input
        x_zero = torch.zeros(4, self.backcast_length)
        forecast = self.model(x_zero)
        self.assertTrue(torch.isfinite(forecast).all())
        # Edge: wrong input shape
        with self.assertRaises(Exception):
            self.model(torch.randn(4, self.backcast_length + 1))

if __name__ == "__main__":
    unittest.main()
