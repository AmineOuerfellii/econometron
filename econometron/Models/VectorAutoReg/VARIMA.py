import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import multivariate_normal, norm
import logging
from econometron.utils.data_preparation.process_timeseries import TimeSeriesProcessor
from econometron.Models.VectorAutoReg import VAR 
from econometron.filters import kalman_objective ,Kalman
from econometron.utils.state_space.update_ss import make_state_space_updater
from econometron.utils.projection import Root
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class VARIMA(VAR):
    def __init__(self, data, max_p=2, max_q=2, d=1, columns=None, criterion='AIC', 
                 forecast_horizon=10, plot=True, bootstrap_n=1000, ci_alpha=0.05, 
                 orth=False, check_stationarity=True, method=None):
        """
        Initialize VARIMA model, inheriting from VAR_.
        
        Parameters:
        -----------
        data : pandas DataFrame
            Input time series data
        max_p : int
            Maximum AR order
        max_q : int
            Maximum MA order
        d : int
            Differencing order
        columns : list
            Columns to process (if None, all columns)
        criterion : str
            Model selection criterion ('AIC' or 'BIC')
        forecast_horizon : int
            Number of periods to forecast
        plot : bool
            Whether to generate plots
        bootstrap_n : int
            Number of bootstrap iterations
        ci_alpha : float
            Confidence interval significance level
        orth : bool
            Whether to use orthogonalized impulse responses
        check_stationarity : bool
            Whether to check stationarity
        method : str
            Transformation method for TimeSeriesProcessor
        """
        super().__init__(data, max_p=max_p, columns=columns, criterion=criterion,
                        forecast_horizon=forecast_horizon, plot=plot, 
                        bootstrap_n=bootstrap_n, ci_alpha=ci_alpha, orth=orth,
                        check_stationarity=check_stationarity, method=method)
        self.max_q = max_q
        self.d = d
        self.tsp = TimeSeriesProcessor(data, columns=columns, method='diff', 
                                     analysis=check_stationarity, plot=plot)
        self.root = Root()
        self.transformed_data = self.tsp.get_transformed_data()
        self.kalman_params = {}
        self.best_q = None
        self.state_space = None
        self.updater = None
        
    def _define_state_space_updater(self, p, q):
        """
        Define state-space updater for VARIMA(p,d,q) using make_state_space_updater.
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
        """
        K = len(self.columns)
        
        def solver(params):
            # F: observation matrix (K x state_dim), P: transition matrix (state_dim x state_dim)
            state_dim = K * max(p, q, 1)
            F = np.zeros((K, state_dim))
            P = np.zeros((state_dim, state_dim))
            
            # AR coefficients
            if p > 0:
                for i in range(p):
                    for j in range(K):
                        for k in range(K):
                            F[j, K * i + k] = params.get(f'phi_{i+1}_{j}_{k}', 0.0)
            
            # MA coefficients
            if q > 0:
                for i in range(q):
                    for j in range(K):
                        for k in range(K):
                            P[K * min(p, i) + j, K * i + k] = params.get(f'theta_{i+1}_{j}_{k}', 0.0)
            
            # State transition for lagged states
            if p > 1 or q > 0:
                for i in range(1, max(p, q)):
                    P[K * (i-1):K * i, K * i:K * (i+1)] = np.eye(K)
            
            return F, P
        
        def build_R(params):
            return np.eye(K)
        
        def build_C(params):
            return np.eye(K)
        
        return make_state_space_updater(
            base_params={},
            solver=solver,
            build_R=build_R,
            build_C=build_C
        )
    
    def _optimize_kalman(self, p, q, y):
        """
        Optimize Kalman filter parameters using kalman_objective and Newton-Raphson.
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
        y : ndarray
            Transformed data (T x K)
        """
        K = len(self.columns)
        param_names = []
        if p > 0:
            param_names += [f'phi_{i+1}_{j}_{k}' for i in range(p) for j in range(K) for k in range(K)]
        if q > 0:
            param_names += [f'theta_{i+1}_{j}_{k}' for i in range(q) for j in range(K) for k in range(K)]
        initial_params = np.zeros(len(param_names)) if param_names else np.array([0.0])
        
        updater = self._define_state_space_updater(p, q)
        
        def objective(params):
            try:
                return kalman_objective(params, {}, param_names, y.T, updater)
            except Exception as e:
                logger.warning(f"Objective evaluation failed: {e}")
                return 1e30
        
        optimized_params, crit = self.root.newton_raphson(
            x0=initial_params,
            func=objective,
            maxit=500,
            stopc=1e-6,
            verbose=False
        )
        
        if crit[0] != 0:
            logger.warning(f"Newton-Raphson did not converge for p={p}, q={q}: crit={crit}")
        
        full_params = {name: val for name, val in zip(param_names, optimized_params)} if param_names else {}
        return full_params, crit, updater
    
    def fit(self, columns=None):
        """
        Fit VARIMA model using SSM and Kalman filter.
        
        Parameters:
        -----------
        columns : list
            Columns to process (if None, all columns)
        """
        if columns is None:
            columns = self.transformed_data.columns
        self.columns = columns
        T, K = self.transformed_data.shape
        
        min_obs = (self.max_p + self.max_q) * K + 1
        if T < min_obs:
            raise ValueError(f"Insufficient observations ({T}) for max_p={self.max_p}, max_q={self.max_q} with {K} variables.")
        
        self.best_criterion_value = float('inf')
        self.all_results = []
        y = self.transformed_data[columns].values
        
        for p in range(0, self.max_p + 1):
            for q in range(0, self.max_q + 1):
                if p == 0 and q == 0:
                    continue  # Skip trivial model
                try:
                    params, crit, updater = self._optimize_kalman(p, q, y)
                    ss_params = updater(params)
                    kalman = Kalman(ss_params)
                    result = kalman.filter(y.T)  # Kalman filter expects (K, T)
                    
                    log_lik = result['log_lik']
                    n_params = K * K * (p + q)
                    aic = -2 * log_lik + 2 * n_params
                    bic = -2 * log_lik + n_params * np.log(T)
                    
                    residuals = (y.T - result['filtered_state'][:K]).T
                    self.all_results.append({
                        'p': p,
                        'q': q,
                        'params': params,
                        'log_lik': log_lik,
                        'aic': aic,
                        'bic': bic,
                        'state_space': ss_params,
                        'kalman_result': result,
                        'residuals': residuals
                    })
                    
                    criterion_value = aic if self.criterion.lower() == 'aic' else bic
                    if criterion_value < self.best_criterion_value:
                        self.best_criterion_value = criterion_value
                        self.best_model = self.all_results[-1]
                        self.best_p = p
                        self.best_q = q
                        self.kalman_params = params
                        self.state_space = ss_params
                        self.updater = updater
                        
                except Exception as e:
                    logger.warning(f"Failed for p={p}, q={q}: {e}")
                    continue
        
        if self.best_model is None:
            raise ValueError("No valid VARIMA model could be fitted")
        
        # Create coefficient table
        self.coeff_table = pd.DataFrame()
        for i, col in enumerate(self.columns):
            for lag in range(self.best_p):
                for j, var in enumerate(self.columns):
                    key = f'phi_{lag+1}_{i}_{j}'
                    self.coeff_table.loc[f'AR_Lag_{lag+1}_{var}', f'{col}_coef'] = self.kalman_params.get(key, 0.0)
            for lag in range(self.best_q):
                for j, var in enumerate(self.columns):
                    key = f'theta_{lag+1}_{i}_{j}'
                    self.coeff_table.loc[f'MA_Lag_{lag+1}_{var}', f'{col}_coef'] = self.kalman_params.get(key, 0.0)
        
        print("\nVARIMA Model Coefficients:")
        print(self.coeff_table)
        
        self.fitted = True
        self.run_full_diagnosis(plot=self.plot, threshold=self.thershold)
        
        if self.plot:
            fitted = self.best_model['kalman_result']['filtered_state'][:K].T
            fitted_df = pd.DataFrame(fitted, index=self.transformed_data.index, columns=self.columns)
            n_vars = K
            n_cols = min(2, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
            axes = np.array(axes).flatten() if n_vars > 1 else [axes]
            for i, col in enumerate(self.columns):
                ax = axes[i]
                ax.plot(self.transformed_data.index, self.transformed_data[col], 'b-', label='Transformed Data')
                ax.plot(fitted_df.index, fitted_df[col], 'r--', label='Fitted Values')
                ax.set_title(f'{col}: Transformed vs Fitted')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            for j in range(n_vars, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.show()
        
        return self.best_model
    
    def predict(self, n_periods=1, plot=True, tol=1e-6):
        """
        Generate forecasts for the VARIMA model.
        """
        if not self.fitted or self.best_model is None:
            raise ValueError("No model fitted. Cannot generate forecasts.")
        
        K = len(self.columns)
        h = n_periods if n_periods > 0 else self.forecast_horizon
        kalman = Kalman(self.state_space)
        y = self.transformed_data[self.columns].values.T
        result = kalman.filter(y)
        state = result['filtered_state'][:, -1]
        
        A = self.state_space['A']
        D = self.state_space['D']
        Q = self.state_space['Q']
        
        forecasts = np.zeros((h, K))
        forecast_vars = np.zeros((h, K))
        current_state = state.copy()
        for t in range(h):
            forecasts[t] = D @ current_state
            forecast_vars[t] = np.diag(D @ Q @ D.T)
            current_state = A @ current_state
        
        se = np.sqrt(forecast_vars)
        ci_lower = forecasts - norm.ppf(1 - self.ci_alpha / 2) * se
        ci_upper = forecasts + norm.ppf(1 - self.ci_alpha / 2) * se
        
        forecast_df = pd.DataFrame(forecasts, columns=self.columns)
        ci_lower_df = pd.DataFrame(ci_lower, columns=self.columns)
        ci_upper_df = pd.DataFrame(ci_upper, columns=self.columns)
        
        if self.d > 0:
            forecast_df = self.tsp.untransform(forecast_df)
            ci_lower_df = self.tsp.untransform(ci_lower_df)
            ci_upper_df = self.tsp.untransform(ci_upper_df)
        
        if isinstance(self.data.index, pd.DatetimeIndex):
            forecast_dates = pd.date_range(
                start=self.data.index[-1] + pd.Timedelta(days=1),
                periods=h,
                freq=self.data.index.freq or 'D'
            )
        else:
            forecast_dates = range(len(self.data), len(self.data) + h)
        
        forecast_df.index = forecast_dates
        ci_lower_df.index = forecast_dates
        ci_upper_df.index = forecast_dates
        
        if plot:
            n_vars = K
            n_cols = min(2, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
            axes = np.array(axes).flatten() if n_vars > 1 else [axes]
            for i, col in enumerate(self.columns):
                ax = axes[i]
                hist_data = self.data[col].iloc[-min(50, len(self.data)):]
                ax.plot(hist_data.index, hist_data.values, 'b-', label='Historical')
                ax.plot(forecast_df.index, forecast_df[col], 'r-', label='Forecast')
                ax.fill_between(forecast_df.index, ci_lower_df[col], ci_upper_df[col],
                               alpha=0.3, color='red', label=f'{100 * (1 - self.ci_alpha)}% CI')
                ax.set_title(f'Forecast for {col}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            for j in range(n_vars, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.show()
        
        return {'point': forecast_df, 'ci_lower': ci_lower_df, 'ci_upper': ci_upper_df}
    
    def impulse_res(self, h=10, orth=True, bootstrap=False, n_boot=1000, plot=False, tol=1e-6):
        """
        Compute impulse response functions (IRFs) for the VARIMA model.
        """
        if not self.fitted or self.best_model is None:
            raise ValueError("No model fitted. Cannot compute IRF.")
        
        K = len(self.columns)
        A = self.state_space['A']
        D = self.state_space['D']
        Q = self.state_space['Q']
        
        Psi = np.zeros((h, K, K))
        Psi[0] = np.eye(K)
        for i in range(1, h):
            Psi[i] = D @ np.linalg.matrix_power(A, i) @ D.T
        
        if orth:
            Sigma = Q[:K, :K]
            P = self._orthogonalize(Sigma)
            irf = np.array([Psi[i] @ P for i in range(h)])
        else:
            irf = Psi
        
        if not bootstrap:
            if plot:
                fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
                axes = axes.flatten() if K > 1 else [axes]
                for i in range(K):
                    for j in range(K):
                        idx = i * K + j
                        axes[idx].plot(range(h), irf[:, i, j], label=f'Shock {self.columns[j]} → {self.columns[i]}')
                        axes[idx].set_title(f'{self.columns[i]} response to {self.columns[j]} shock')
                        axes[idx].set_xlabel('Horizon')
                        axes[idx].set_ylabel('Response')
                        axes[idx].grid(True)
                        axes[idx].legend()
                plt.tight_layout()
                plt.show()
            return irf
        
        boot_irfs = np.zeros((n_boot, h, K, K))
        y = self.transformed_data[self.columns].values
        T = y.shape[0]
        
        for b in range(n_boot):
            boot_idx = np.random.choice(T, size=T, replace=True)
            boot_y = y[boot_idx]
            try:
                params, _, updater = self._optimize_kalman(self.best_p, self.best_q, boot_y)
                ss_params = updater(params)
                boot_A = ss_params['A']
                boot_D = ss_params['D']
                boot_Q = ss_params['Q']
                
                boot_Psi = np.zeros((h, K, K))
                boot_Psi[0] = np.eye(K)
                for i in range(1, h):
                    boot_Psi[i] = boot_D @ np.linalg.matrix_power(boot_A, i) @ boot_D.T
                
                if orth:
                    boot_Sigma = boot_Q[:K, :K]
                    P = self._orthogonalize(boot_Sigma)
                    boot_irf = np.array([boot_Psi[i] @ P for i in range(h)])
                else:
                    boot_irf = boot_Psi
                boot_irfs[b] = boot_irf
            except Exception as e:
                logger.warning(f"Bootstrap iteration {b} failed: {e}")
                continue
        
        ci_lower = np.percentile(boot_irfs, 100 * self.ci_alpha / 2, axis=0)
        ci_upper = np.percentile(boot_irfs, 100 * (1 - self.ci_alpha / 2), axis=0)
        
        if plot:
            fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
            axes = axes.flatten() if K > 1 else [axes]
            for i in range(K):
                for j in range(K):
                    idx = i * K + j
                    axes[idx].plot(range(h), irf[:, i, j], label=f'Shock {self.columns[j]} → {self.columns[i]}')
                    axes[idx].fill_between(range(h), ci_lower[:, i, j], ci_upper[:, i, j], 
                                          alpha=0.3, color='red', label=f'{100 * (1 - self.ci_alpha)}% CI')
                    axes[idx].set_title(f'{self.columns[i]} response to {self.columns[j]} shock')
                    axes[idx].set_xlabel('Horizon')
                    axes[idx].set_ylabel('Response')
                    axes[idx].grid(True)
                    axes[idx].legend()
            plt.tight_layout()
            plt.show()
        
        return irf, ci_lower, ci_upper
    
    def FEVD(self, h=10, plot=False):
        """
        Compute Forecast Error Variance Decomposition (FEVD).
        """
        K = len(self.columns)
        irf = self.impulse_res(h=h, orth=True, bootstrap=False, plot=False)
        Sigma = self.state_space['Q'][:K, :K]
        
        fevd = np.zeros((h, K, K))
        mse = np.zeros((h, K))
        for i in range(h):
            for j in range(K):
                for t in range(i + 1):
                    mse[i, j] += np.sum(irf[t, j, :] ** 2 * np.diag(Sigma))
                for k in range(K):
                    fevd[i, j, k] = np.sum(irf[:i + 1, j, k] ** 2 * Sigma[k, k]) / mse[i, j] if mse[i, j] != 0 else 0
        
        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for j in range(K):
                bottom = np.zeros(h)
                for k in range(K):
                    axes[j].bar(range(h), fevd[:, j, k], bottom=bottom, label=f'Shock from {self.columns[k]}')
                    bottom += fevd[:, j, k]
                axes[j].set_title(f'FEVD for {self.columns[j]}')
                axes[j].set_xlabel('Horizon')
                axes[j].set_ylabel('Variance Contribution')
                axes[j].legend()
                axes[j].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return fevd
    
    def simulate(self, n_periods=100, plot=False, tol=1e-6):
        """
        Simulate time series using the fitted VARIMA model.
        """
        K = len(self.columns)
        A = self.state_space['A']
        D = self.state_space['D']
        Q = self.state_space['Q']
        
        Y_sim = np.zeros((n_periods, K))
        state = np.zeros(A.shape[0])
        state[:K] = self.transformed_data.values[-1]
        
        for t in range(n_periods):
            Y_sim[t] = D @ state
            state = A @ state + multivariate_normal.rvs(mean=np.zeros(K), cov=Q[:K, :K])
        
        Y_sim_df = pd.DataFrame(Y_sim, columns=self.columns)
        if self.d > 0:
            Y_sim_df = self.tsp.untransform(Y_sim_df)
        
        if isinstance(self.data.index, pd.DatetimeIndex):
            sim_dates = pd.date_range(
                start=self.data.index[-1] + pd.Timedelta(days=1),
                periods=n_periods,
                freq=self.data.index.freq or 'D'
            )
        else:
            sim_dates = range(len(self.data), len(self.data) + n_periods)
        Y_sim_df.index = sim_dates
        
        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for i in range(K):
                axes[i].plot(Y_sim_df.index, Y_sim_df[self.columns[i]], label=f'Simulated {self.columns[i]}')
                axes[i].set_title(f'Simulated Series for {self.columns[i]}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True)
            plt.tight_layout()
            plt.show()
        
        return Y_sim_df