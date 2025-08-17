from econometron.utils.estimation.Regression import ols_estimator
from .VAR import VAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVAR(VAR):
    def __init__(self, data, max_p=2, columns=None, criterion='AIC', forecast_horizon=10,
                 plot=True, bootstrap_n=1000, ci_alpha=0.05, orth=True, check_stationarity=True,
                 method=None, Threshold=0.8):
        super().__init__(data, max_p, columns, criterion, forecast_horizon, plot,
                         bootstrap_n, ci_alpha, orth, check_stationarity, method, Threshold)
        self.A = None
        self.B = None
        self.A_inv_B = None
        self.identification_method = None
        self.structural_shocks = None

    def identify(self, method='chol', A=None, B=None):
        """
        Define and store A, B, and A⁻¹B matrices for structural identification.

        Parameters:
        - method: str, 'chol' (Cholesky), 'blanchard-quah' (Blanchard-Quah), or 'AB' (AB model)
        - A: np.ndarray, A matrix for AB model (optional)
        - B: np.ndarray, B matrix for AB model (optional)
        """
        if self.best_model is None:
            raise ValueError(
                "No model fitted. Cannot perform structural identification.")

        K = len(self.columns)
        Sigma = np.cov(self.best_model['residuals'].T)

        if method not in ['chol', 'blanchard-quah', 'AB']:
            raise ValueError(
                "Identification method must be 'chol', 'blanchard-quah', or 'AB'")

        self.identification_method = method

        if method == 'chol':
            # Cholesky decomposition: A is lower triangular, B is identity
            self.B = np.eye(K)
            self.A = np.linalg.cholesky(Sigma+1e-8*np.eye(K))
            self.A_inv_B = self.A
        elif method == 'blanchard-quah':
            # Blanchard-Quah: long-run restrictions
            # Compute cumulative IRF at long horizon
            irf = self.impulse_res(
                h=50, orth=False, bootstrap=False, plot=False)
            long_run_effect = irf[-1]  # Last horizon for long-run effect
            # QR decomposition to ensure orthogonality
            Q, R = np.linalg.qr(long_run_effect)
            self.A = np.eye(K)
            self.B = Q
            self.A_inv_B = Q
        elif method == 'AB':
            # AB model: user-specified A and B matrices
            if A is None or B is None:
                raise ValueError(
                    "A and B matrices must be provided for AB identification")
            if A.shape != (K, K) or B.shape != (K, K):
                raise ValueError(
                    "A and B must be square matrices of size K x K")
            self.A = A
            self.B = B
            self.A_inv_B = solve(self.A, self.B, assume_a='gen')
            # Verify identification: A * Sigma * A' = B * B'
            if not np.allclose(self.A @ Sigma @ self.A.T, self.B @ self.B.T):
                raise ValueError(
                    "AB model does not satisfy identification condition")

        self.structural_shocks = self.get_structural_shocks()

    def _orthogonalize(self, Sigma):
        """
        Override _orthogonalize to use structural identification.
        """
        if self.A_inv_B is None:
            raise ValueError(
                "Structural identification not performed. Run identify() first.")
        return self.A_inv_B

    def impulse_res(self, h=10, orth=True, bootstrap=False, n_boot=1000, plot=False, tol=1e-6):
        """
        Override impulse_res to compute structural IRF.
        """
        if self.best_model is None:
            raise ValueError("No model fitted. Cannot compute IRF.")
        if self.A_inv_B is None and orth:
            raise ValueError(
                "Structural identification not performed. Run identify() first.")

        K = len(self.columns)
        p = self.best_model['p']
        beta = self.best_model['beta']
        intercept_included = beta.shape[0] == K * p + 1
        A = beta[1:] if intercept_included else beta
        A = A.reshape(p, K, K).transpose(2, 1, 0)
        Psi = np.zeros((h+1, K, K))
        Psi[0] = np.eye(K)
        for i in range(1, h+1):
            for j in range(min(i, p)):
                Psi[i] += A[:, :, j] @ Psi[i-j-1]
        if orth:
            irf = np.array([Psi[i] @ self.A_inv_B for i in range(h+1)])
        else:
            irf = Psi
        if not bootstrap:
            if plot:
                fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
                axes = axes.flatten() if K > 1 else [axes]
                for i in range(K):
                    for j in range(K):
                        idx = i * K + j
                        axes[idx].plot(range(h+1), irf[:, i, j],
                                       label=f'Shock {self.columns[j]} → {self.columns[i]}')
                        axes[idx].set_title(
                            f'Structural IRF: {self.columns[i]} response to {self.columns[j]} shock')
                        axes[idx].set_xlabel('Horizon')
                        axes[idx].set_ylabel('Response')
                        axes[idx].grid(True)
                        axes[idx].legend()
                plt.tight_layout()
                plt.show()
            return irf

        boot_irfs = np.zeros((n_boot, h, K, K))
        residuals = self.best_model['residuals']
        T, K = residuals.shape
        data = self.data.values

        for b in range(n_boot):
            boot_idx = np.random.choice(T, size=T, replace=True)
            boot_resids = residuals[boot_idx]
            Y_sim = np.zeros((T + p, K))
            Y_sim[:p] = np.flipud(data[-p:])
            intercept = beta[0] if intercept_included else np.zeros(K)

            for t in range(p, T + p):
                Y_t = intercept.copy()
                for j in range(p):
                    Y_t += A[:, :, j] @ Y_sim[t - j - 1]
                Y_t += boot_resids[t - p]
                Y_sim[t] = Y_t
            Y_sim = Y_sim[p:]
            X, Y = self.lag_matrix(p)

            try:
                boot_beta, _, _, _ = ols_estimator(X, Y_sim, tol=tol)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {b} failed: {e}")
                continue

            boot_A = boot_beta[1:] if boot_beta.shape[0] == K * \
                p + 1 else boot_beta
            boot_A = boot_A.reshape(p, K, K).transpose(2, 1, 0)
            boot_Psi = np.zeros((h+1, K, K))
            boot_Psi[0] = np.eye(K)
            for i in range(1, h+1):
                for j in range(min(i, p)):
                    boot_Psi[i] += boot_A[:, :, j] @ boot_Psi[i - j - 1]

            if orth:
                boot_Sigma = np.cov(boot_resids.T)
                try:
                    if self.identification_method == 'chol':
                        P = np.linalg.cholesky(boot_Sigma)
                    elif self.identification_method == 'blanchard-quah':
                        boot_irf_long = np.array(
                            [boot_Psi[i] for i in range(50)])
                        Q, _ = np.linalg.qr(boot_irf_long[-1])
                        P = Q
                    else:  # AB model
                        P = solve(self.A, self.B, assume_a='gen')
                    boot_irf = np.array([boot_Psi[i] @ P for i in range(h+1)])
                except np.linalg.LinAlgError:
                    logger.warning(
                        f"Bootstrap iteration {b} failed: Non-positive definite covariance")
                    continue
            else:
                boot_irf = boot_Psi
            boot_irfs[b] = boot_irf

        ci_lower = np.percentile(boot_irfs, 100 * self.ci_alpha / 2, axis=0)
        ci_upper = np.percentile(
            boot_irfs, 100 * (1 - self.ci_alpha / 2), axis=0)

        if plot:
            fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
            axes = axes.flatten() if K > 1 else [axes]
            for i in range(K):
                for j in range(K):
                    idx = i * K + j
                    axes[idx].plot(range(h+1), irf[:, i, j],
                                   label=f'Shock {self.columns[j]} → {self.columns[i]}')
                    axes[idx].fill_between(range(h+1), ci_lower[:, i, j], ci_upper[:, i, j],
                                           alpha=0.3, color='red',
                                           label=f'{100 * (1 - self.ci_alpha)}% CI')
                    axes[idx].set_title(
                        f'Structural IRF: {self.columns[i]} response to {self.columns[j]} shock')
                    axes[idx].set_xlabel('Horizon')
                    axes[idx].set_ylabel('Response')
                    axes[idx].grid(True)
                    axes[idx].legend()
            plt.tight_layout()
            plt.show()

        return irf, ci_lower, ci_upper

    def FEVD(self, h=10, plot=False):
        """
        Override FEVD to use structural IRF.
        """
        K = len(self.columns)
        irf = self.impulse_res(h=h, orth=True, bootstrap=False, plot=False)
        Sigma = np.cov(self.best_model['residuals'].T)
        fevd = np.zeros((h+1, K, K))
        mse = np.zeros((h+1, K))

        for i in range(h+1):
            for j in range(K):
                for t in range(i + 1):
                    mse[i, j] += np.sum(irf[t, j, :] ** 2 *
                                        np.diag(self.B @ self.B.T))
                for k in range(K):
                    fevd[i, j, k] = np.sum(
                        irf[:i + 1, j, k] ** 2 * (self.B @ self.B.T)[k, k]) / mse[i, j] if mse[i, j] != 0 else 0

        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for j in range(K):
                bottom = np.zeros(h+1)
                for k in range(K):
                    axes[j].bar(range(h+1), fevd[:, j, k], bottom=bottom,
                                label=f'Structural Shock from {self.columns[k]}')
                    bottom += fevd[:, j, k]
                axes[j].set_title(f'Structural FEVD for {self.columns[j]}')
                axes[j].set_xlabel('Horizon')
                axes[j].set_ylabel('Variance Contribution')
                axes[j].legend()
                axes[j].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return fevd

    def get_structural_shocks(self):
        """
        Compute structural shocks from reduced-form residuals.
        Returns: np.ndarray of structural shocks
        """
        if self.A is None or self.best_model is None:
            raise ValueError(
                "Structural identification not performed or model not fitted.")

        residuals = self.best_model['residuals']
        structural_shocks = solve(self.A, residuals.T, assume_a='gen').T
        return structural_shocks

    def shock_decomposition(self, h=10, plot=False):
        """
        Compute shock decomposition of the series.
        Returns: np.ndarray of shape (T, K, K+1) where last dimension includes contribution of each shock plus initial conditions
        """
        if self.best_model is None or self.A_inv_B is None:
            raise ValueError(
                "Model not fitted or structural identification not performed.")

        K = len(self.columns)
        T = self.data.shape[0]
        p = self.best_model['p']
        beta = self.best_model['beta']
        intercept_included = beta.shape[0] == K * p + 1
        A = beta[1:] if intercept_included else beta
        A = A.reshape(p, K, K).transpose(2, 1, 0)

        # Compute structural shocks
        structural_shocks = self.get_structural_shocks()

        # Compute IRFs
        Psi = np.zeros((h+1, K, K))
        Psi[0] = np.eye(K)
        for i in range(1, h+1):
            for j in range(min(i, p)):
                Psi[i] += A[:, :, j] @ Psi[i - j - 1]
        structural_irf = np.array([Psi[i] @ self.A_inv_B for i in range(h+1)])

        # Initialize decomposition
        decomp = np.zeros((T, K, K + 1))  # +1 for initial conditions
        data = self.data.values
        intercept = beta[0] if intercept_included else np.zeros(K)

        # Compute contributions
        for t in range(p, T):
            # Initial conditions (mean effect and lagged values)
            initial = np.zeros(K)
            for j in range(p):
                initial += A[:, :, j] @ data[t - j - 1]
            if intercept_included:
                initial += intercept
            decomp[t, :, K] = initial

            # Shock contributions
            for k in range(K):
                for s in range(min(t - p + 1, h+1)):
                    decomp[t, :, k] += structural_irf[s, :, k] * \
                        structural_shocks[t - s - p, k]

        # Fill in initial periods with zeros
        decomp[:p, :, :] = 0

        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for j in range(K):
                bottom = np.zeros(T)
                for k in range(K + 1):
                    label = f'Shock from {self.columns[k]}' if k < K else 'Initial Conditions'
                    axes[j].plot(range(T), decomp[:, j, k], label=label)
                    axes[j].plot(range(T), self.data[self.columns[j]],
                                 'k--', label='Actual Data')
                axes[j].set_title(f'Shock Decomposition for {self.columns[j]}')
                axes[j].set_xlabel('Time')
                axes[j].set_ylabel('Value')
                axes[j].legend()
                axes[j].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return decomp
