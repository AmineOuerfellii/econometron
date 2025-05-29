import pandas as pd
import numpy as np
from scipy.stats import chi2, shapiro
from statsmodels.tsa.stattools import acf, adfuller
import matplotlib.pyplot as plt
from itertools import product
from utils.data_preparation import process_time_series
from utils.estimation.OLS import ols_estimator

class VAR:
    """
    Vector Autoregression (VAR) model class with OLS estimation.
    """
    def __init__(self, max_p=2, criterion='AIC', max_diff=2, significance_level=0.05, forecast_horizon=5, plot=True):
        """
        Initialize VAR model parameters.
        """
        self.max_p = max_p
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
        self.best_criterion_value = None
        self.stationarity_results = None
        self.all_results = None
        self.forecasts = None
        self.residual_diag_results = None
        self.coefficient_table = None
    
    def create_lag_matrix(self, data, lags):
        """
        Create lagged variable matrix for VAR model.
        """
        T, K = data.shape
        X = np.ones((T - lags, 1))
        for lag in range(1, lags + 1):
            lag_data = data[lags-lag:T-lag]
            if lag_data.ndim == 1:
                lag_data = lag_data.reshape(-1, 1)
            X = np.hstack((X, lag_data))
        Y = data[lags:]
        return X, Y
    
    def compute_aic_bic(self, Y, residuals, K, p, T):
        """
        Compute AIC and BIC for model evaluation.
        """
        resid_cov = np.cov(residuals.T)
        log_det = np.log(np.linalg.det(resid_cov + 1e-10 * np.eye(K)))
        n_params = K * (K * p + 1)
        aic = T * log_det + 2 * n_params
        bic = T * log_det + n_params * np.log(T)
        return aic, bic
    
    def forecast(self, data, beta, p, h):
        """
        Generate h-step-ahead forecasts with confidence intervals.
        """
        T, K = data.shape
        forecasts = np.zeros((h, K))
        forecast_vars = np.zeros((h, K))
        last_observations = data[-p:].copy()
        
        resid_cov = np.cov(self.best_model['residuals'].T) if self.best_model else np.eye(K) * 1e-10
        
        for t in range(h):
            X_t = np.ones((1, 1))
            for lag in range(p):
                lag_data = last_observations[-(lag+1)]
                if lag_data.ndim == 1:
                    lag_data = lag_data.reshape(1, -1)
                X_t = np.hstack((X_t, lag_data))
            forecast_t = X_t @ beta
            forecasts[t] = forecast_t
            
            forecast_vars[t] = np.diag(resid_cov) * (t + 1)
            
            last_observations = np.vstack((last_observations[1:], forecast_t))
        
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
        Fit the VAR model using OLS with grid search.
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
        
        min_observations = self.max_p * K + 1
        if T < min_observations:
            raise ValueError(f"Insufficient observations ({T}) for max_p={self.max_p} with {K} variables. Need at least {min_observations}.")
        
        self.best_criterion_value = float('inf')
        self.all_results = []
        
        for p in range(1, self.max_p + 1):
            try:
                X, Y = self.create_lag_matrix(data_array, p)
                
                beta, residuals, se, z_values, p_values = ols_estimator(X, Y)
                
                aic, bic = self.compute_aic_bic(Y, residuals, K, p, Y.shape[0])
                crit_value = aic if self.criterion == 'AIC' else bic
                
                self.all_results.append({
                    'p': p,
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
                        'fitted': X @ beta,
                        'residuals': residuals,
                        'se': se,
                        'z_values': z_values,
                        'p_values': p_values
                    }
                    self.best_p = p
                
            except Exception as e:
                print(f"Failed for p={p}: {str(e)}")
                continue
        
        if self.best_model is None:
            raise ValueError("No valid VAR model could be fitted. Check data or reduce max_p.")
        
        self.forecasts = self.forecast(data_array, self.best_model['beta'], self.best_p, self.forecast_horizon)
        
        self.residual_diag_results = self.residual_diagnostics(self.best_model['residuals'], columns)
        
        self.coefficient_table = pd.DataFrame()
        for k, col in enumerate(columns):
            for lag in range(self.best_p):
                for j, var in enumerate(columns):
                    idx = 1 + lag * K + j
                    self.coefficient_table.loc[f'Lag_{lag+1}_{var}', f'{col}_coef'] = self.best_model['beta'][idx, k]
                    self.coefficient_table.loc[f'Lag_{lag+1}_{var}', f'{col}_se'] = self.best_model['se'][idx, k]
                    self.coefficient_table.loc[f'Lag_{lag+1}_{var}', f'{col}_z'] = self.best_model['z_values'][idx, k]
                    self.coefficient_table.loc[f'Lag_{lag+1}_{var}', f'{col}_p'] = self.best_model['p_values'][idx, k]
            self.coefficient_table.loc['Constant', f'{col}_coef'] = self.best_model['beta'][0, k]
            self.coefficient_table.loc['Constant', f'{col}_se'] = self.best_model['se'][0, k]
            self.coefficient_table.loc['Constant', f'{col}_z'] = self.best_model['z_values'][0, k]
            self.coefficient_table.loc['Constant', f'{col}_p'] = self.best_model['p_values'][0, k]
        
        if self.plot:
            for i, col in enumerate(columns):
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
        
        print(f"\nBest VAR Model:")
        print(f"Lags: {self.best_p}")
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
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting.")
        
        h = h or self.forecast_horizon
        forecasts = self.forecast(self.model_data.to_numpy(), self.best_model['beta'], self.best_p, h)
        forecast_dates = pd.date_range(start=self.model_data.index[-1] + pd.offsets.MonthEnd(1), 
                                     periods=h, freq=self.model_data.index.freq)
        
        forecast_df = pd.DataFrame(forecasts['point'], index=forecast_dates, columns=self.columns)
        for col in self.columns:
            col_idx = self.columns.index(col)
            forecast_df[f'{col}_ci_lower'] = forecasts['ci_lower'][:, col_idx]
            forecast_df[f'{col}_ci_upper'] = forecasts['ci_upper'][:, col_idx]
        
        return forecast_df
