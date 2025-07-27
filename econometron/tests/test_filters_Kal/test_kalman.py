import unittest
import numpy as np
from unittest.mock import Mock
from econometron.filters import Kalman, kalman_objective, kalman_smooth

class TestKalman(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with valid Kalman filter parameters and data."""
        self.valid_params = {
            'A': np.array([[0.5, 0.0], [0.0, 0.5]]),  # 2x2 state transition matrix
            'D': np.array([[1.0, 0.0], [0.0, 1.0]]),  # 2x2 observation matrix
            'Q': np.array([[0.1, 0.0], [0.0, 0.1]]),  # 2x2 state covariance
            'R': np.array([[0.01, 0.0], [0.0, 0.01]]),  # 2x2 observation covariance
            'x0': np.array([[0.0], [0.0]]),  # 2x1 initial state
            'P0': np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 initial covariance
        }
        self.y = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])  # 2x3 observations
        self.n = 2  # State dimension
        self.m = 2  # Observation dimension
        self.T = 3  # Time steps

    # Tests for Kalman class initialization
    def test_kalman_init_valid(self):
        """Test Kalman initialization with valid parameters."""
        kalman = Kalman(self.valid_params)
        self.assertTrue(np.array_equal(kalman.A, self.valid_params['A']))
        self.assertTrue(np.array_equal(kalman.D, self.valid_params['D']))
        self.assertTrue(np.array_equal(kalman.Q, self.valid_params['Q']))
        self.assertTrue(np.array_equal(kalman.R, self.valid_params['R']))
        self.assertTrue(np.array_equal(kalman.X_0, self.valid_params['x0']))
        self.assertTrue(np.array_equal(kalman.P_0, self.valid_params['P0']))
        self.assertEqual(kalman.n, self.n)
        self.assertEqual(kalman.m, self.m)

    def test_kalman_init_invalid_A_shape(self):
        """Test initialization with invalid A matrix shape."""
        invalid_params = self.valid_params.copy()
        invalid_params['A'] = np.array([[1.0, 0.0]])  # Not square
        with self.assertRaises(ValueError):
            Kalman(invalid_params)

    def test_kalman_init_invalid_D_shape(self):
        """Test initialization with invalid D matrix shape."""
        invalid_params = self.valid_params.copy()
        invalid_params['D'] = np.array([[1.0]])  # Wrong shape
        with self.assertRaises(ValueError):
            Kalman(invalid_params)

    def test_kalman_init_invalid_x0_shape(self):
        """Test initialization with invalid x0 shape."""
        invalid_params = self.valid_params.copy()
        invalid_params['x0'] = np.array([1.0, 0.0])  # Not column vector
        with self.assertRaises(ValueError):
            Kalman(invalid_params)

    def test_kalman_init_invalid_P0_shape(self):
        """Test initialization with invalid P0 shape."""
        invalid_params = self.valid_params.copy()
        invalid_params['P0'] = np.array([[1.0]])  # Wrong shape
        with self.assertRaises(ValueError):
            Kalman(invalid_params)

    # Tests for _compute_initial_covariance
    def test_compute_initial_covariance_provided(self):
        """Test _compute_initial_covariance with provided P0."""
        kalman = Kalman(self.valid_params)
        P0 = self.valid_params['P0']
        result = kalman._compute_initial_covariance(P0)
        self.assertTrue(np.array_equal(result, P0))

    def test_compute_initial_covariance_stationary(self):
        """Test _compute_initial_covariance for stationary system."""
        params = self.valid_params.copy()
        params.pop('P0', None)  # No P0 provided
        kalman = Kalman(params)
        P0 = kalman._compute_initial_covariance(None)
        self.assertEqual(P0.shape, (self.n, self.n))
        self.assertTrue(np.allclose(P0, P0.T))  # Check symmetry

    def test_compute_initial_covariance_non_stationary(self):
        """Test _compute_initial_covariance for non-stationary system."""
        params = self.valid_params.copy()
        params['A'] = np.array([[1.0, 0.0], [0.0, 1.0]])  # Non-stationary
        params.pop('P0', None)
        kalman = Kalman(params)
        P0 = kalman._compute_initial_covariance(None)
        self.assertTrue(np.array_equal(P0, np.eye(self.n) * 1e6))

    # Tests for filter method
    def test_filter_valid(self):
        """Test filter method with valid observations."""
        kalman = Kalman(self.valid_params)
        result = kalman.filter(self.y)
        self.assertEqual(result['x_tt'].shape, (self.n, self.T))
        self.assertEqual(result['P_tt'].shape, (self.n, self.n, self.T))
        self.assertEqual(result['x_tt1'].shape, (self.n, self.T))
        self.assertEqual(result['P_tt1'].shape, (self.n, self.n, self.T))
        self.assertEqual(result['residuals'].shape, (self.m, self.T))
        self.assertIsInstance(result['log_lik'], float)
        self.assertFalse(np.any(np.isnan(result['x_tt'])))
        self.assertFalse(np.any(np.isnan(result['P_tt'])))

    def test_filter_invalid_y_shape(self):
        """Test filter with invalid observation shape."""
        kalman = Kalman(self.valid_params)
        invalid_y = np.array([[1.0, 1.1]])  # Wrong number of rows
        with self.assertRaises(ValueError):
            kalman.filter(invalid_y)

    def test_filter_singular_omega(self):
        """Test filter with singular Omega matrix."""
        params = self.valid_params.copy()
        params['R'] = np.ones((self.m, self.m))
        kalman = Kalman(params)
        result = kalman.filter(self.y)
        self.assertEqual(result['log_lik'], -8e30)

    # Tests for smooth method
    def test_smooth_valid(self):
        """Test smooth method with valid observations."""
        kalman = Kalman(self.valid_params)
        result = kalman.smooth(self.y)
        self.assertEqual(result['Xsm'].shape, (self.n, self.T))
        self.assertEqual(result['Xtilde'].shape, (self.n, self.T))
        self.assertEqual(result['PP1'].shape, (self.n, self.n, self.T))
        self.assertEqual(result['residuals'].shape, (self.m, self.T))
        self.assertFalse(np.any(np.isnan(result['Xsm'])))

    # Tests for kalman_objective function
    def test_kalman_objective_valid(self):
        """Test kalman_objective with valid inputs."""
        update_state_space = Mock(return_value=self.valid_params)
        params = np.array([0.5, 0.1])  # Example parameters
        fixed_params = {'D': self.valid_params['D'], 'R': self.valid_params['R']}
        param_names = ['A', 'Q']
        result = kalman_objective(params, fixed_params, param_names, self.y, update_state_space)
        self.assertIsInstance(result, float)
        self.assertFalse(np.isnan(result))

    def test_kalman_objective_error(self):
        """Test kalman_objective with invalid parameters."""
        update_state_space = Mock(side_effect=ValueError("Invalid params"))
        params = np.array([0.5, 0.1])
        fixed_params = {'D': self.valid_params['D'], 'R': self.valid_params['R']}
        param_names = ['A', 'Q']
        result = kalman_objective(params, fixed_params, param_names, self.y, update_state_space)
        self.assertEqual(result, 8e30)  # Expected penalty

    # Tests for kalman_smooth function
    def test_kalman_smooth_valid(self):
        """Test kalman_smooth with valid inputs."""
        update_state_space = Mock(return_value=self.valid_params)
        result = kalman_smooth(self.y, update_state_space)
        self.assertEqual(result.shape, (self.n, self.T))
        self.assertFalse(np.any(np.isnan(result)))

    def test_kalman_smooth_error(self):
        """Test kalman_smooth with invalid parameters."""
        update_state_space = Mock(side_effect=ValueError("Invalid params"))
        result = kalman_smooth(self.y, update_state_space)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()