import pandas as pd
import numpy as np
from scipy.stats import chi2, shapiro
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from itertools import product
from utils.data_preparation import process_time_series

class VARIMA:
    """
    Vector Autoregressive Integrated Moving Average (VARIMA) model class using MLE with gradient-based optimization.
    """
    def __init__(self, max_p=2, max_q=2, criterion='AIC', max_diff=2, significance_level=0.05, forecast_horizon=5, plot=True):
        """
        Initialize VARIMA model parameters.
        
        Parameters:
        - max_p: Maximum AR lag order.
        - max_q: Maximum MA lag order.
        - criterion: Model selection criterion ('AIC' or 'BIC').
        - max_diff: Maximum differencing order for stationarity.
        - significance_level: Significance level for stationarity tests.
        - forecast_horizon: Number of steps for forecasting.
        - plot: Whether to generate diagnostic plots.
        """
        self.max_p = max_p
        self.max_q = max_q
        self.criterion = criterion.upper()
        self.max_diff = max_diff
        self.significance_level = significance_level
        self.forecast_horizon = forecast_horizon
        self.plot = plot
        self.fitted = False
        self.model_data = None
        self.columns = None
        self.diff_orders = None
        self.best_model = None
        self.best_p = None
        self.best_q = None
        self.best_criterion_value = None
        self.stationarity_results = None
        self.all_results = None
        self.forecasts = None
        self.residual_diag_results = None
        self.coefficient_table = None
    
    def create_lag_matrix(self, data, p, q, residuals=None):
        """
        Create lagged variable and residual matrices for VARIMA model.
        
        Parameters:
        - data: Differenced time series data (numpy array).
        - p: AR lag order.
        - q: MA lag order.
        - residuals: Residuals for MA terms (if available).
        
        Returns:
        - X: Design matrix with lagged variables and residuals.
        - Y: Target matrix.
        """
        T, K = data.shape
        max_lag = max(p, q)
        X = np.ones((T - max_lag, 1))  # Constant term
        for lag in range(1, p + 1):
            lag_data = data[max_lag-lag:T-lag]
            if lag_data.ndim == 1:
                lag_data = lag_data.reshape(-1, 1)
            X = np.hstack((X, lag_data))
        
        if q > 0 and residuals is not None:
            for lag in range(1, q + 1):
                lag_resid = residuals[max_lag-lag:T-lag]
                if lag_resid.ndim == 1:
                    lag_resid = lag_resid.reshape(-1, 1)
                X = np.hstack((X, lag_resid))
        
        Y = data[max_lag:]
        return X, Y
    
    def compute_residuals(self, data, params, p, q, K):
        """
        Compute residuals for given parameters.
        
        Parameters:
        - data: Differenced time series data.
        - params: Parameter vector (constant, AR, MA).
        - p: AR lag order.
        - q: MA lag order.
        - K: Number of variables.
        
        Returns:
        - residuals: Computed residuals.
        """
        T = data.shape[0]
        max_lag = max(p, q)
        residuals = np.zeros((T - max_lag, K))
        
        # Reshape parameters
        beta = params.reshape(-1, K)
        
        # Initialize residuals for early time points
        past_residuals = np.zeros((max_lag, K))
        
        for t in range(T - max_lag):
            X_t = np.ones((1, 1))
            for lag in range(1, p + 1):
                lag_data = data[max_lag-lag+t] if t >= lag else np.zeros(K)
                if lag_data.ndim == 1:
                    lag_data = lag_data.reshape(1, -1)
                X_t = np.hstack((X_t, lag_data))
            for lag in range(1, q + 1):
                lag_resid = past_residuals[max_lag-lag] if t >= lag else np.zeros(K)
                if lag_resid.ndim == 1:
                    lag_resid = lag_resid.reshape(1, -1)
                X_t = np.hstack((X_t, lag_resid))
            
            forecast_t = X_t @ beta
            residuals[t] = data[max_lag+t] - forecast_t
            if q > 0:
                past_residuals = np.vstack((past_residuals[1:], residuals[t]))
        
        return residuals
    
    def log_likelihood(self, params, data, p, q, K):
        """
        Compute negative log-likelihood for MLE.
        
        Parameters:
        - params: Parameter vector (constant, AR, MA).
        - data: Differenced time series data.
        - p: AR lag order.
        - q: MA lag order.
        - K: Number of variables.
        
        Returns:
        - Negative log-likelihood.
        """
        residuals = self.compute_residuals(data, params, p, q, K)
        resid_cov = np.cov(residuals.T) + 1e-10 * np.eye(K)
        log_det = np.log(np.linalg.det(resid_cov))
        T = residuals.shape[0]
        return 0.5 * T * log_det
    
    def compute_aic_bic(self, Y, residuals, K, p, q, T):
        """
        Compute AIC and BIC for model evaluation.
        
        Parameters:
        - Y: Target data.
        - residuals: Model residuals.
        - K: Number of variables.
        - p: AR lag order.
        - q: MA lag order.
        - T: Number of observations.
        
        Returns:
        - aic: Akaike Information Criterion.
        - bic: Bayesian Information Criterion.
        """
        resid_cov = np.cov(residuals.T) + 1e-10 * np.eye(K)
        log_det = np.log(np.linalg.det(resid_cov))
        n_params = K * (K * p + K * q + 1)  # Constant, AR, MA
        aic = T * log_det + 2 * n_params
        bic = T * log_det + n_params * np.log(T)
        return aic, bic
    
    def forecast(self, data, beta, p, q, residuals, h):
        """
        Generate h-step-ahead forecasts with confidence intervals.
        
        Parameters:
        - data: Differenced time series data.
        - beta: Estimated coefficients (AR and MA).
        - p: AR lag order.
        - q: MA lag order.
        - residuals: Residuals for MA terms.
        - h: Forecast horizon.
        
        Returns:
        - Dictionary with point forecasts, lower and upper confidence intervals.
        """
        T, K = data.shape
        forecasts = np.zeros((h, K))
        forecast_vars = np.zeros((h, K))
        last_observations = data[-p:].copy() if p > 0 else np.zeros((1, K))
        last_residuals = residuals[-q:].copy() if q > 0 else np.zeros((1, K))
        
        resid_cov = np.cov(residuals.T) if residuals is not None else np.eye(K) * 1e-10
        
        for t in range(h):
            X_t = np.ones((1, 1))
            for lag in range(p):
                lag_data = last_observations[-(lag+1)] if lag < len(last_observations) else np.zeros(K)
                if lag_data.ndim == 1:
                    lag_data = lag_data.reshape(1, -1)
                X_t = np.hstack((X_t, lag_data))
            for lag in range(q):
                lag_resid = last_residuals[-(lag+1)] if lag < len(last_residuals) else np.zeros(K)
                if lag_resid.ndim == 1:
                    lag_resid = lag_resid.reshape(1, -1)
                X_t = np.hstack((X_t, lag_resid))
            
            forecast_t = X_t @ beta
            forecasts[t] = forecast_t
            forecast_vars[t] = np.diag(resid_cov) * (t + 1)
            
            last_observations = np.vstack((last_observations[1:], forecast_t)) if p > 0 else last_observations
            last_residuals = np.vstack((last_residuals[1:], np.zeros(K))) if q > 0 else last_residuals
        
        se = np.sqrt(forecast_vars)
        ci_lower = forecasts - 1.96 * se
        ci_upper = forecasts + 1.96 * se
        
        return {
            'point': forecasts,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def residual_diagnostics(self, residuals, columns):
        """
        Perform residual diagnostics with Ljung-Box and Shapiro-Wilk tests.
        
        Parameters:
        - residuals: Model residuals.
        - columns: Variable names.
        
        Returns:
        - Dictionary with diagnostic results.
        """
        diagnostics = {}
        T = residuals.shape[0]
        
        for i, col in enumerate(columns):
            resid = residuals[:, i]
            acf_vals = acf(resid, nlags=10, fft=False)
            lb_stat = T * (T + 2) * sum(acf_vals[k]**2 / (T - k) for k in range(1, 11))
            lb_pvalue = 1 - chi2.cdf(lb_stat, df=10)
            sw_stat, sw_pvalue = shapiro(resid)
            
            diagnostics[col] = {
                'mean': np.mean(resid),
                'variance': np.var(resid),
                'acf': acf_vals,
                'ljung_box': {'statistic': lb_stat, 'p_value': lb_pvalue},
                'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pvalue}
            }
            
            if self.plot:
                plt.figure(figsize=(12, 4))
                plt.subplot(121)
                plt.plot(resid)
                plt.title(f'Residuals for {col}')
                plt.xlabel('Time')
                plt.ylabel('Residual')
                plt.subplot(122)
                plt.stem(acf_vals)
                plt.title(f'Residual ACF for {col}')
                plt.xlabel('Lag')
                plt.ylabel('ACF')
                plt.tight_layout()
                plt.show()
        
        return diagnostics
    
    def fit(self, data, date_column=None, columns=None):
        """
        Fit the VARIMA model using MLE with grid search over p and q.
        
        Parameters:
        - data: Input time series data (pandas DataFrame or Series).
        - date_column: Column name for dates (if any).
        - columns: Variables to include in the model.
        
        Returns:
        - Self (fitted model).
        """
        if self.criterion not in ['AIC', 'BIC']:
            raise ValueError("criterion must be 'AIC' or 'BIC'")
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        self.columns = columns
        
        self.stationarity_results = process_time_series(data, date_column, columns, 
                                                      self.max_diff, self.significance_level, plot=False)
        
        self.model_data = pd.DataFrame({col: self.stationarity_results[col].get('differenced', 
                                                                               self.stationarity_results[col]['original']) 
                                       for col in columns}).dropna()
        self.diff_orders = {col: self.stationarity_results[col].get('diff_order', 0) for col in columns}
        
        data_array = self.model_data.to_numpy()
        T, K = data_array.shape
        
        min_observations = max(self.max_p, self.max_q) * K + 1
        if T < min_observations:
            raise ValueError(f"Insufficient observations ({T}) for max_p={self.max_p}, max_q={self.max_q} with {K} variables. Need at least {min_observations}.")
        
        self.best_criterion_value = float('inf')
        self.all_results = []
        
        for p, q in product(range(self.max_p + 1), range(self.max_q + 1)):
            if p == 0 and q == 0:
                continue
            try:
                # Initialize parameters: constant (K), AR (p * K * K), MA (q * K * K)
                n_params = K * (1 + p * K + q * K)
                initial_params = np.random.randn(n_params) * 0.01
                
                # Optimize negative log-likelihood
                result = minimize(self.log_likelihood, initial_params, args=(data_array, p, q, K),
                                method='BFGS', options={'disp': False})
                
                if not result.success:
                    print(f"Optimization failed for p={p}, q={q}: {result.message}")
                    continue
                
                beta = result.x.reshape(-1, K)
                residuals = self.compute_residuals(data_array, result.x, p, q, K)
                
                # Compute standard errors (approximate via Hessian)
                hess_inv = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(n_params)
                se = np.sqrt(np.diag(hess_inv)).reshape(-1, K)
                z_values = beta / (se + 1e-10)
                p_values = 2 * (1 - chi2.cdf(np.abs(z_values)**2, df=1))
                
                # Compute AIC/BIC
                aic, bic = self.compute_aic_bic(residuals, residuals, K, p, q, T)
                crit_value = aic if self.criterion == 'AIC' else bic
                
                self.all_results.append({
                    'p': p,
                    'q': q,
                    'beta': beta,
                    'residuals': residuals,
                    'se': se,
                    'z_values': z_values,
                    'p_values': p_values,
                    'aic': aic,
                    'bic': bic
                })
                
                if crit_value < self.best_criterion_value:
                    self.best_criterion_value = crit_value
                    self.best_model = {
                        'beta': beta,
                        'fitted': self.create_lag_matrix(data_array, p, q, residuals)[0] @ beta,
                        'residuals': residuals,
                        'se': se,
                        'z_values': z_values,
                        'p_values': p_values
                    }
                    self.best_p = p
                    self.best_q = q
                
            except Exception as e:
                print(f"Failed for p={p}, q={q}: {str(e)}")
                continue
        
        if self.best_model is None:
            raise ValueError("No valid VARIMA model could be fitted. Check data or reduce max_p/max_q.")
        
        self.forecasts = self.forecast(data_array, self.best_model['beta'], self.best_p, self.best_q, 
                                     self.best_model['residuals'], self.forecast_horizon)
        
        self.residual_diag_results = self.residual_diagnostics(self.best_model['residuals'], self.columns)
        
        self.coefficient_table = pd.DataFrame()
        for k, col in enumerate(self.columns):
            idx = 0
            self.coefficient_table.loc['Constant', f'{col}_coef'] = self.best_model['beta'][idx, k]
            self.coefficient_table.loc['Constant', f'{col}_se'] = self.best_model['se'][idx, k]
            self.coefficient_table.loc['Constant', f'{col}_z'] = self.best_model['z_values'][idx, k]
            self.coefficient_table.loc['Constant', f'{col}_p'] = self.best_model['p_values'][idx, k]
            idx += 1
            for lag in range(self.best_p):
                for j, var in enumerate(self.columns):
                    self.coefficient_table.loc[f'AR_Lag_{lag+1}_{var}', f'{col}_coef'] = self.best_model['beta'][idx, k]
                    self.coefficient_table.loc[f'AR_Lag_{lag+1}_{var}', f'{col}_se'] = self.best_model['se'][idx, k]
                    self.coefficient_table.loc[f'AR_Lag_{lag+1}_{var}', f'{col}_z'] = self.best_model['z_values'][idx, k]
                    self.coefficient_table.loc[f'AR_Lag_{lag+1}_{var}', f'{col}_p'] = self.best_model['p_values'][idx, k]
                    idx += 1
            for lag in range(self.best_q):
                for j, var in enumerate(self.columns):
                    self.coefficient_table.loc[f'MA_Lag_{lag+1}_{var}', f'{col}_coef'] = self.best_model['beta'][idx, k]
                    self.coefficient_table.loc[f'MA_Lag_{lag+1}_{var}', f'{col}_se'] = self.best_model['se'][idx, k]
                    self.coefficient_table.loc[f'MA_Lag_{lag+1}_{var}', f'{col}_z'] = self.best_model['z_values'][idx, k]
                    self.coefficient_table.loc[f'MA_Lag_{lag+1}_{var}', f'{col}_p'] = self.best_model['p_values'][idx, k]
                    idx += 1
        
        if self.plot:
            for i, col in enumerate(self.columns):
                plt.figure(figsize=(10, 4))
                plt.plot(self.model_data.index[-len(self.best_model['fitted']):], 
                        self.model_data[col][-len(self.best_model['fitted']):], 
                        label='Observed', alpha=0.7)
                plt.plot(self.model_data.index[-len(self.best_model['fitted']):], 
                        self.best_model['fitted'][:, i], 
                        label='Fitted', linestyle='--')
                plt.title(f'Observed vs Fitted for {col}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.tight_layout()
                plt.show()
        
        print(f"\nBest VARIMA Model:")
        print(f"AR Lags: {self.best_p}, MA Lags: {self.best_q}")
        print(f"{self.criterion}: {self.best_criterion_value:.4f}")
        for col, diff_order in self.diff_orders.items():
            print(f"{col} differencing order: {diff_order}")
        print("\nCoefficient Table:")
        print(self.coefficient_table.round(4))
        print("\nResidual Diagnostics:")
        for col, diag in self.residual_diag_results.items():
            print(f"{col}:")
            print(f"  Mean = {diag['mean']:.4f}, Variance = {diag['variance']:.4f}")
            print(f"  Ljung-Box Test: Statistic = {diag['ljung_box']['statistic']:.4f}, p-value = {diag['ljung_box']['p_value']:.4f}")
            print(f"  Shapiro-Wilk Test: Statistic = {diag['shapiro_wilk']['statistic']:.4f}, p-value = {diag['shapiro_wilk']['p_value']:.4f}")
        
        self.fitted = True
        return self
    
    def predict(self, h=None):
        """
        Generate forecasts with confidence intervals using the fitted model.
        
        Parameters:
        - h: Forecast horizon (defaults to self.forecast_horizon).
        
        Returns:
        - DataFrame with point forecasts and confidence intervals.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting.")
        
        h = h or self.forecast_horizon
        forecasts = self.forecast(self.model_data.to_numpy(), self.best_model['beta'], 
                                self.best_p, self.best_q, self.best_model['residuals'], h)
        forecast_dates = pd.date_range(start=self.model_data.index[-1] + pd.offsets.MonthEnd(1), 
                                     periods=h, freq=self.model_data.index.freq)
        
        forecast_df = pd.DataFrame(forecasts['point'], index=forecast_dates, columns=self.columns)
        for col in self.columns:
            col_idx = self.columns.index(col)
            forecast_df[f'{col}_ci_lower'] = forecasts['ci_lower'][:, col_idx]
            forecast_df[f'{col}_ci_upper'] = forecasts['ci_upper'][:, col_idx]
        
        return forecast_df