import numpy as np
from scipy.linalg import inv, solve_discrete_lyapunov, eigvals
import scipy.linalg as sp
import warnings

class Kalman:
    """
    Kalman filter and smoother for state-space models.

    Parameters:
    -----------
    params : dict
        Dictionary containing:
        - A: State transition matrix (n x n).
        - D: Observation matrix (m x n).
        - Q: State covariance matrix (n x n).
        - R: Observation covariance matrix (m x m).
        - x0: Initial state estimate (n x 1, optional).
        - P0: Initial state covariance (n x n, optional).
    """
    def __init__(self, params):
        """Initialize the Kalman filter with model parameters."""
        # Convert inputs to numpy arrays
        self.A = np.array(params['A'])
        self.D = np.array(params['D'])
        self.Q = np.array(params['Q'])
        self.R = np.array(params['R'])

        # Dimensions
        self.n = self.A.shape[0]  # State dimension
        self.m = self.D.shape[0]  # Observation dimension

        # Validate shapes
        if self.A.shape != (self.n, self.n):
            raise ValueError("A must be square (n x n).")
        if self.Q.shape != (self.n, self.n):
            raise ValueError("Q must be square (n x n).")
        if self.D.shape != (self.m, self.n):
            raise ValueError("D must have shape (m x n).")
        if self.R.shape != (self.m, self.m):
            raise ValueError("R must be square (m x m).")

        # # Validate symmetry
        # if not np.allclose(self.Q, self.Q.T):
        #     raise ValueError("Q must be symmetric.")
        # if not np.allclose(self.R, self.R.T):
        #     raise ValueError("R must be symmetric.")

        # # Validate positive definiteness
        # if np.any(eigvals(self.Q) < -1e-10):
        #     raise ValueError("Q must be positive semi-definite.")
        # if np.any(eigvals(self.R) < -1e-10):
        #     raise ValueError("R must be positive definite.")

        # Initial state
        self.X_0 = np.array(params.get('x0', np.zeros((self.n, 1))))
        if self.X_0.shape != (self.n, 1):
            raise ValueError("x0 must be a column vector (n x 1).")

        # Initial covariance
        self.P_0 = self._compute_initial_covariance(params.get('P0'))

    def _compute_initial_covariance(self, P0):
        """
        Compute initial state covariance P_0.

        If P0 is provided, use it. Otherwise, for stationary systems, solve the
        discrete Lyapunov equation P0 = A P0 A^T + Q. For non-stationary systems,
        use a large diagonal covariance.

        Args:
            P0: Initial covariance matrix (n x n, optional)

        Returns:
            P0: Initial covariance matrix (n x n)
        """
        if P0 is not None:
            P0 = np.array(P0)
            if P0.shape != (self.n, self.n):
                raise ValueError("P0 must have shape (n x n).")
            if not np.allclose(P0, P0.T):
                raise ValueError("P0 must be symmetric.")
            if np.any(eigvals(P0) < -1e-10):
                raise ValueError("P0 must be positive semi-definite.")
            return P0

        # Check stationarity
        eigenvalues = eigvals(self.A)
        if np.any(np.abs(eigenvalues) >= 1):
            return np.eye(self.n) * 1e6  # Large covariance for non-stationary system
        # Solve P0 = A P0 A^T + Q
        try:
            P0 = solve_discrete_lyapunov(self.A, self.Q)
            if not np.allclose(P0, P0.T):
                P0 = (P0 + P0.T) / 2  # Ensure symmetry
            return P0
        except np.linalg.LinAlgError:
            return np.eye(self.n) * 1e6

    def filter(self, y):
        """
        Run Kalman filter using the core algorithm from the provided code.

        Parameters:
        -----------
        y : ndarray
            Observations (m x T).

        Returns:
        --------
        dict
            - x_tt: Filtered states (n x T).
            - P_tt: Filtered covariances (n x n x T).
            - x_tt1: Predicted states (n x T).
            - P_tt1: Predicted covariances (n x n x T).
            - residuals: Standardized residuals (m x T).
            - log_lik: Log-likelihood.
        """
        if y.shape[0] != self.m:
            raise ValueError(f"Observations dimension {y.shape[0]} does not match D rows {self.m}")
        T = y.shape[1]
        x_tt = np.zeros((self.n, T))
        P_tt = np.zeros((self.n, self.n, T))
        x_tt1 = np.zeros((self.n, T))
        P_tt1 = np.zeros((self.n, self.n, T))
        residuals = np.zeros((self.m, T))
        log_lik = 0

        # Initialize
        x_t = self.X_0
        P_t = self.P_0
        # Core Kalman filter loop
        for t in range(T):
            Ztilde = y[:, [t]] - self.D @ x_t
            # Update
            Omega = self.D @ P_t@ self.D.T + self.R
            # Check for NaNs/Infs in Omega
            if np.any(np.isnan(Omega)) or np.any(np.isinf(Omega)):
                warnings.warn(f"NaN/Inf detected in innovation covariance Omega at time step {t}.", stacklevel=2)
                # Assign a large penalty and break or skip
                log_lik = - 8e30
                break # Exit the loop
            try:
              Omegainv = np.linalg.inv(Omega)
            except np.linalg.LinAlgError:
                warnings.warn(f"Could not invert innovation covariance Omega at time step {t}.", stacklevel=2)
                log_lik= - 8e30
                break
            Kt = P_t @ self.D.T @ Omegainv
            x_t=self.A @ x_t + self.A @ Kt @ Ztilde
            P_t= self.A@ ( P_t - P_t @ self.D.T @ Omegainv @ self.D @ P_t) @ self.A.T + self.Q
            P_tt[:, :, t] = P_t
            residuals[:, t] = Ztilde.T @ inv(sp.sqrtm(Omega))
            # Log-likelihood contribution
            log_lik = log_lik -0.5 * (np.log(2*np.pi)+np.log(np.linalg.det(Omega))) -0.5*Ztilde.conj().T@Omegainv@Ztilde
        if isinstance(log_lik, (list, np.ndarray)):
            log_lik = np.mean(log_lik)
        # Handle log-likelihood
        if log_lik.imag != 0:
            log_lik = - 8e30
        else:
            log_lik = -log_lik.real

        return {
            'x_tt': x_tt,
            'P_tt': P_tt,
            'x_tt1': x_tt1,
            'P_tt1': P_tt1,
            'residuals': residuals,
            'log_lik': log_lik
        }

    def smooth(self, y):
        """
        Kalman smoother (with D only, no D2).

        Parameters:
        -----------
        y : ndarray
            Observations (m x T)

        Returns:
        --------
        dict:
            - Xsm: Smoothed states (n x T)
            - Xtilde: Predicted states (n x T)
            - PP1: Predicted covariances (n x n x T)
            - residuals: Innovations / residuals (m x T)
        """
        T = y.shape[1]
        dimX = self.n
        dimZ = self.m

        # Initial state covariance
        CC = self.Q
        if np.max(np.abs(np.linalg.eigvals(self.A))) >= 1:
            P0 = CC * 1000
        else:
            P0 = np.linalg.solve(np.eye(dimX**2) - np.kron(self.A, self.A), CC.flatten()).reshape(dimX, dimX)

        # Init
        Xhat = np.zeros((dimX, T + 1))
        PP0 = np.zeros((dimX, dimX, T))
        PP1 = np.zeros((dimX, dimX, T))
        residuals = np.zeros((dimZ, T))

        D = self.D

        # Forward recursion
        for t in range(T):
            Ztilde = y[:, t] - D @ self.A @ Xhat[:, t]
            Omega = D @ self.A @ P0 @ (D @ self.A).T + D @ self.Q @ D.T + self.R
            Omega_inv = np.linalg.inv(Omega + np.eye(dimZ) * 1e-8)
            K = (self.A @ P0 @ D.T + self.Q @ D.T) @ Omega_inv
            Xhat[:, t + 1] = self.A @ Xhat[:, t] + K @ Ztilde
            P1 = self.A @ P0 @ self.A.T + self.Q
            P0 = P1 - K @ Omega @ K.T
            PP0[:, :, t] = P0
            PP1[:, :, t] = self.A @ P0 @ self.A.T + self.Q
            residuals[:, t] = Ztilde

        # Prediction for smoother
        Xtilde = self.A @ Xhat[:, :-1]
        Xsm = np.zeros_like(Xhat)
        Xsm[:, T] = Xhat[:, T]

        # Backward recursion
        for t in range(T - 1, -1, -1):
            J = PP0[:, :, t] @ self.A.T @ np.linalg.inv(PP1[:, :, t] + np.eye(dimX) * 1e-8)
            Xsm[:, t] = Xhat[:, t] + J @ (Xsm[:, t + 1] - Xtilde[:, t])

            
        Xsm = Xsm[:, :-1]

        return {
            'Xsm': Xsm[:, :T],
            'Xtilde': Xtilde,
            'PP1': PP1,
            'residuals': residuals
        }