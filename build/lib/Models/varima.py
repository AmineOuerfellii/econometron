import pandas as pd
import numpy as np
from scipy.stats import chi2, shapiro
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.interpolate import interp1d
from pandas.tseries.frequencies import to_offset

def process_time_series(data, date_column=None, columns=None, max_diff=2, significance_level=0.05, plot=True):
    """
    Process time series data: handle missing values, check stationarity, and visualize ACF/PACF.
    
    Parameters:
    - data: DataFrame or Series with time series data
    - date_column: str, name of the date column (if DataFrame)
    - columns: list, columns to analyze (if None, all numeric columns are used)
    - max_diff: int, maximum differencing order
    - significance_level: float, p-value threshold for ADF test
    - plot: bool, whether to plot ACF/PACF
    
    Returns:
    - dict: Results including stationarity status, differenced data, and ADF results
    """
    
    # Convert to DataFrame if Series
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Set index if date_column is provided
    if date_column:
        data = data.set_index(date_column)
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
    
    # Select columns
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    # Infer frequency if not set
    if data.index.inferred_freq is None:
        try:
            data.index.freq = pd.infer_freq(data.index)
        except:
            print("Could not infer frequency. Trying common frequencies...")
            common_freqs = ['M', 'Q', 'A', 'D']
            for freq in common_freqs:
                try:
                    data.index = pd.date_range(start=data.index[0], periods=len(data), freq=freq)
                    data.index.freq = freq
                    break
                except:
                    continue
            if data.index.inferred_freq is None:
                print("Setting to monthly frequency as fallback.")
                data.index = pd.date_range(start=data.index[0], periods=len(data), freq='M')
    
    results = {}
    
    for col in columns:
        print(f"\nProcessing column: {col}")
        series = data[col].copy()
        
        # Handle missing values
        if series.isna().any():
            print(f"Found {series.isna().sum()} missing values in {col}")
            # Interpolate for internal missing values
            series = series.interpolate(method='linear', limit_direction='both')
            
            # Extrapolate if missing at ends
            if series.isna().any():
                non_na_idx = series.dropna().index
                if len(non_na_idx) > 1:
                    f = interp1d(non_na_idx.map(lambda x: x.timestamp()), series.dropna(), 
                               fill_value='extrapolate')
                    series = pd.Series(f(series.index.map(lambda x: x.timestamp())), 
                                     index=series.index)
        
        # Initialize result dictionary for this column
        results[col] = {'original': series, 'adf_results': {}, 'stationary': False}
        
        # ADF test on original series
        adf_result = adfuller(series, autolag='AIC')
        results[col]['adf_results'][0] = {
            'p_value': adf_result[1],
            'statistic': adf_result[0],
            'critical_values': adf_result[4]
        }
        
        if adf_result[1] < significance_level:
            print(f"{col} is stationary (p-value: {adf_result[1]:.4f})")
            results[col]['stationary'] = True
        else:
            print(f"{col} is not stationary (p-value: {adf_result[1]:.4f})")
            # Try differencing
            for diff_order in range(1, max_diff + 1):
                diff_series = series.diff(diff_order).dropna()
                adf_result = adfuller(diff_series, autolag='AIC')
                results[col]['adf_results'][diff_order] = {
                    'p_value': adf_result[1],
                    'statistic': adf_result[0],
                    'critical_values': adf_result[4]
                }
                if adf_result[1] < significance_level:
                    print(f"{col} becomes stationary after {diff_order} differencing (p-value: {adf_result[1]:.4f})")
                    results[col]['stationary'] = True
                    results[col]['differenced'] = diff_series
                    results[col]['diff_order'] = diff_order  # Store differencing order
                    break
                else:
                    results[col]['differenced'] = diff_series
        
        # ACF and PACF plots
        if plot and results[col]['stationary']:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(121)
            plot_acf(results[col].get('differenced', series), lags=10, ax=plt.gca())
            plt.title(f'ACF - {col}')
            
            plt.subplot(122)
            plot_pacf(results[col].get('differenced', series), lags=10, ax=plt.gca())
            plt.title(f'PACF - {col}')
            
            plt.tight_layout()
            plt.show()
    
    return results

class VARIMA:
    """
    Vector Autoregressive Integrated Moving Average (VARIMA) model class using MLE with gradient-based optimization.
    Assumes input data is stationary, as provided by the process_time_series function.
    """
    def __init__(self, max_p=2, max_q=2, criterion='AIC', max_diff=2, significance_level=0.05, forecast_horizon=5, plot=True):
        """
        Initialize VARIMA model parameters.
        
        Parameters:
        - max_p: Maximum AR lag order (non-negative integer).
        - max_q: Maximum MA lag order (non-negative integer).
        - criterion: Model selection criterion ('AIC' or 'BIC').
        - max_diff: Maximum differencing order for stationarity (non-negative integer).
        - significance_level: Significance level for stationarity tests (between 0 and 1).
        - forecast_horizon: Number of steps for forecasting (positive integer).
        - plot: Whether to generate diagnostic plots (boolean).
        """
        if max_p < 0 or max_q < 0:
            raise ValueError("max_p and max_q must be non-negative")
        if max_diff < 0:
            raise ValueError("max_diff must be non-negative")
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if criterion.upper() not in ['AIC', 'BIC']:
            raise ValueError("criterion must be 'AIC' or 'BIC'")
        
        self.max_p = max_p
        self.max_q = max_q
        self.criterion = criterion.upper()
        self.max_diff = max_diff
        self.significance_level = significance_level
        self.forecast_horizon = forecast_horizon
        self.plot = plot
        self.fitted = False
        self.model_data = None
        self.original_data = None
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
        max_lag = max(p, q, 1)  # Ensure at least 1 lag
        if T <= max_lag:
            raise ValueError(f"Data length {T} is too short for max_lag {max_lag}")
        
        X = np.ones((T - max_lag, 1))  # Constant term
        
        # Add AR terms
        for lag in range(1, p + 1):
            lag_data = data[max_lag-lag:T-lag]
            if lag_data.shape[0] != T - max_lag:
                lag_data = np.zeros((T - max_lag, K))
            X = np.hstack((X, lag_data))
        
        # Add MA terms
        if q > 0 and residuals is not None:
            for lag in range(1, q + 1):
                if residuals.shape[0] >= max_lag:
                    lag_resid = residuals[max_lag-lag:T-lag]
                    if lag_resid.shape[0] != T - max_lag:
                        lag_resid = np.zeros((T - max_lag, K))
                else:
                    lag_resid = np.zeros((T - max_lag, K))
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
        max_lag = max(p, q, 1)
        if T <= max_lag:
            return np.zeros((T - max_lag, K))
        
        residuals = np.zeros((T - max_lag, K))
        beta = params.reshape(-1, K)
        past_residuals = np.zeros((max_lag, K))
        
        for t in range(T - max_lag):
            X_t = np.ones((1, 1))
            
            # Add AR terms
            for lag in range(1, p + 1):
                lag_data = data[max_lag - lag + t] if max_lag - lag + t >= 0 else np.zeros(K)
                X_t = np.hstack((X_t, lag_data.reshape(1, -1)))
            
            # Add MA terms
            for lag in range(1, q + 1):
                lag_resid = past_residuals[max_lag - lag] if t >= lag else np.zeros(K)
                X_t = np.hstack((X_t, lag_resid.reshape(1, -1)))
            
            # Compute forecast and residual
            if X_t.shape[1] == beta.shape[0]:
                forecast_t = X_t @ beta
                residuals[t] = data[max_lag + t] - forecast_t.flatten()
            else:
                residuals[t] = np.zeros(K)
            
            # Update past residuals
            if q > 0:
                past_residuals = np.vstack((past_residuals[1:], residuals[t].reshape(1, -1)))
        
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
        try:
            residuals = self.compute_residuals(data, params, p, q, K)
            if residuals.shape[0] <= K:
                return 1e10
            
            resid_cov = np.cov(residuals.T) + 1e-3 * np.eye(K)  # Increased regularization
            if not np.all(np.linalg.eigvals(resid_cov) > 0):
                return 1e10
            
            log_det = np.log(np.linalg.det(resid_cov))
            T = residuals.shape[0]
            
            ll = -0.5 * T * (K * np.log(2 * np.pi) + log_det)
            resid_cov_inv = np.linalg.pinv(resid_cov)  # Use pseudo-inverse
            for t in range(T):
                ll -= 0.5 * residuals[t] @ resid_cov_inv @ residuals[t]
            
            return -ll
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return 1e10
    
    def compute_aic_bic(self, residuals, K, p, q, T):
        """
        Compute AIC and BIC for model evaluation.
        
        Parameters:
        - residuals: Model residuals.
        - K: Number of variables.
        - p: AR lag order.
        - q: MA lag order.
        - T: Number of observations.
        
        Returns:
        - aic: Akaike Information Criterion.
        - bic: Bayesian Information Criterion.
        """
        try:
            resid_cov = np.cov(residuals.T) + 1e-3 * np.eye(K)  # Increased regularization
            if not np.all(np.linalg.eigvals(resid_cov) > 0):
                return np.inf, np.inf
            
            log_det = np.log(np.linalg.det(resid_cov))
            n_params = K * (1 + K * p + K * q)
            
            ll = -0.5 * T * (K * np.log(2 * np.pi) + log_det)
            for t in range(T):
                ll -= 0.5 * residuals[t] @ np.linalg.pinv(resid_cov) @ residuals[t]
            
            aic = -2 * ll + 2 * n_params
            bic = -2 * ll + n_params * np.log(T)
            
            return aic, bic
        except (np.linalg.LinAlgError, ValueError):
            return np.inf, np.inf
    
    def inverse_difference(self, series, forecasts, diff_order, initial_values):
        """
        Inverse differencing to transform forecasts back to original scale.
        
        Parameters:
        - series: Original series (before differencing).
        - forecasts: Forecasted values (differenced scale).
        - diff_order: Order of differencing applied.
        - initial_values: Last values of the original series for integration.
        
        Returns:
        - Inverse differenced forecasts.
        """
        if diff_order == 0:
            return forecasts
        
        result = forecasts.copy()
        if len(initial_values) == 0:
            initial_values = [0]
        
        if diff_order == 1:
            result = np.cumsum(result) + initial_values[-1]
        else:
            for d in range(diff_order):
                result = np.cumsum(result) + (initial_values[-1] if len(initial_values) > 0 else 0)
        
        return result
    
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
        
        last_observations = data[-max(p, 1):].copy() if p > 0 else np.zeros((1, K))
        last_residuals = residuals[-max(q, 1):].copy() if q > 0 and residuals is not None else np.zeros((1, K))
        
        resid_cov = np.eye(K) * 0.01
        if residuals is not None and residuals.shape[0] > K:
            resid_cov = np.cov(residuals.T) + 1e-4 * np.eye(K)
        
        for t in range(h):
            X_t = np.ones((1, 1))
            for lag in range(1, p + 1):
                lag_data = last_observations[-(lag)] if lag <= len(last_observations) else np.zeros(K)
                X_t = np.hstack((X_t, lag_data.reshape(1, -1)))
            
            for lag in range(1, q + 1):
                lag_resid = last_residuals[-(lag)] if t == 0 and lag <= len(last_residuals) else np.zeros(K)
                X_t = np.hstack((X_t, lag_resid.reshape(1, -1)))
            
            if X_t.shape[1] == beta.shape[0]:
                forecasts[t] = (X_t @ beta).flatten()
            else:
                forecasts[t] = np.zeros(K)
            
            forecast_vars[t] = np.diag(resid_cov) * np.sqrt(t + 1)
            
            if p > 0:
                last_observations = np.vstack((last_observations[1:], forecasts[t].reshape(1, -1)))
            if q > 0:
                last_residuals = np.vstack((last_residuals[1:], np.zeros((1, K))))
        
        se = np.sqrt(forecast_vars)
        ci_lower = forecasts - 1.96 * se
        ci_upper = forecasts + 1.96 * se
        
        undiff_forecasts = np.zeros_like(forecasts)
        undiff_ci_lower = np.zeros_like(ci_lower)
        undiff_ci_upper = np.zeros_like(ci_upper)
        
        for i, col in enumerate(self.columns):
            diff_order = self.diff_orders.get(col, 0)
            initial_values = self.original_data[col].values[-max(1, diff_order):]
            undiff_forecasts[:, i] = self.inverse_difference(
                self.original_data[col], forecasts[:, i], diff_order, initial_values
            )
            undiff_ci_lower[:, i] = self.inverse_difference(
                self.original_data[col], ci_lower[:, i], diff_order, initial_values
            )
            undiff_ci_upper[:, i] = self.inverse_difference(
                self.original_data[col], ci_upper[:, i], diff_order, initial_values
            )
        
        return {
            'point': undiff_forecasts,
            'ci_lower': undiff_ci_lower,
            'ci_upper': undiff_ci_upper
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
            max_lags = max(1, min(10, T // 4))
            
            try:
                acf_vals = acf(resid, nlags=max_lags, fft=False)
                lb_stat = T * (T + 2) * sum(acf_vals[k]**2 / (T - k) for k in range(1, max_lags + 1))
                lb_pvalue = 1 - chi2.cdf(lb_stat, df=max_lags)
                
                sw_stat, sw_pvalue = np.nan, np.nan
                if 3 <= len(resid) <= 5000:
                    sw_stat, sw_pvalue = shapiro(resid)
                
                diagnostics[col] = {
                    'mean': np.mean(resid),
                    'variance': np.var(resid),
                    'acf': acf_vals,
                    'ljung_box': {'statistic': lb_stat, 'p_value': lb_pvalue},
                    'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pvalue}
                }
            except Exception as e:
                print(f"Warning: Diagnostics failed for {col}: {e}")
                diagnostics[col] = {
                    'mean': np.mean(resid),
                    'variance': np.var(resid),
                    'acf': np.array([1.0]),
                    'ljung_box': {'statistic': np.nan, 'p_value': np.nan},
                    'shapiro_wilk': {'statistic': np.nan, 'p_value': np.nan}
                }
            
            if self.plot:
                try:
                    plt.figure(figsize=(12, 4))
                    plt.subplot(121)
                    plt.plot(resid)
                    plt.title(f'Residuals for {col}')
                    plt.xlabel('Time')
                    plt.ylabel('Residual')
                    
                    plt.subplot(122)
                    plt.stem(range(len(diagnostics[col]['acf'])), diagnostics[col]['acf'])
                    plt.title(f'Residual ACF for {col}')
                    plt.xlabel('Lag')
                    plt.ylabel('ACF')
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Warning: Could not plot diagnostics for {col}: {e}")
        
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
      
      # Convert Series to DataFrame
      if isinstance(data, pd.Series):
          data = data.to_frame()
      
      # Select columns
      if columns is None:
          columns = list(data.select_dtypes(include=[np.number]).columns)
      self.columns = columns
      
      # Store original data for inverse differencing
      self.original_data = data[columns].copy()
      
      # Set datetime index for original data
      if date_column and date_column in data.columns:
          self.original_data = self.original_data.set_index(data[date_column])
          if not pd.api.types.is_datetime64_any_dtype(self.original_data.index):
              self.original_data.index = pd.to_datetime(self.original_data.index)
      elif not pd.api.types.is_datetime64_any_dtype(self.original_data.index):
          self.original_data.index = pd.date_range(start='2000-01-01', periods=len(self.original_data), freq='M')
      
      # Process time series for stationarity
      print("Processing time series for stationarity...")
      self.stationarity_results = process_time_series(
          data, date_column, columns, self.max_diff, self.significance_level, plot=self.plot
      )
      
      # Create model data with differenced series
      model_series = {}
      self.diff_orders = {}
      
      for col in columns:
          col_result = self.stationarity_results[col]
          if 'differenced' in col_result:
              model_series[col] = col_result['differenced']
              self.diff_orders[col] = col_result.get('diff_order', 1)
          else:
              model_series[col] = col_result['original']
              self.diff_orders[col] = 0
      
      self.model_data = pd.DataFrame(model_series).dropna()
      
      # Ensure model_data has a datetime index and infer frequency
      if not pd.api.types.is_datetime64_any_dtype(self.model_data.index):
          if len(self.model_data) <= len(self.original_data):
              self.model_data.index = self.original_data.index[-len(self.model_data):]
          else:
              self.model_data.index = pd.date_range(
                  start=self.original_data.index[0], 
                  periods=len(self.model_data), 
                  freq=self.original_data.index.freq or 'M'
              )
      
      if self.model_data.index.freq is None:
          try:
              inferred_freq = pd.infer_freq(self.model_data.index)
              if inferred_freq is None:
                  inferred_freq = 'M'
              self.model_data.index = pd.date_range(
                  start=self.model_data.index[0], 
                  periods=len(self.model_data), 
                  freq=inferred_freq
              )
          except:
              self.model_data.index = pd.date_range(
                  start=self.model_data.index[0], 
                  periods=len(self.model_data), 
                  freq='M'
              )
      
      # Convert to numpy array for model fitting
      data_array = self.model_data.to_numpy()
      T, K = data_array.shape
      
      # Check minimum observations requirement
      min_observations = max(self.max_p, self.max_q, 1) * K + K + 1
      if T < min_observations:
          raise ValueError(f"Insufficient observations ({T}) for max_p={self.max_p}, max_q={self.max_q}, and {K} variables. Need at least {min_observations}.")
      
      print(f"Fitting VARIMA model with {T} observations and {K} variables...")
      
      # Grid search for best p and q
      self.best_criterion_value = float('inf')
      self.all_results = []
      
      for p, q in product(range(self.max_p + 1), range(self.max_q + 1)):
          if p == 0 and q == 0:
              continue
          
          try:
              print(f"Trying VARIMA({p}, d, {q})...")
              
              # Calculate number of parameters
              n_params = K * (1 + p * K + q * K)
              
              # Check if the number of parameters is reasonable
              if n_params > T // 2:
                  print(f"  Too many parameters ({n_params}) for {T} observations. Skipping...")
                  continue
              
              # Initialize parameters
              np.random.seed(42)
              initial_params = np.random.randn(n_params) * 0.01
              
              # Optimize parameters
              result = minimize(
                  self.log_likelihood, 
                  initial_params, 
                  args=(data_array, p, q, K),
                  method='L-BFGS-B', 
                  bounds=[(-1, 1)] * n_params,
                  options={'disp': False, 'maxiter': 1000}
              )
              
              if not result.success:
                  print(f"  Optimization failed: {result.message}")
                  continue
              
              # Reshape parameters
              beta = result.x.reshape(-1, K)
              
              # Compute residuals
              residuals = self.compute_residuals(data_array, result.x, p, q, K)
              
              if residuals.shape[0] <= K:
                  print(f"  Insufficient residuals for covariance estimation")
                  continue
              
              # Compute standard errors
              try:
                  hess_inv = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv
                  se = np.sqrt(np.abs(np.diag(hess_inv))).reshape(-1, K)
                  z_values = beta / (se + 1e-10)
                  p_values = 2 * (1 - chi2.cdf(np.abs(z_values)**2, df=1))
              except:
                  se = np.ones_like(beta) * 0.01
                  z_values = beta / se
                  p_values = np.ones_like(beta) * 0.5
              
              # Compute AIC and BIC
              aic, bic = self.compute_aic_bic(residuals, K, p, q, residuals.shape[0])
              
              if np.isfinite(aic) and np.isfinite(bic):
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
                  
                  print(f"  {self.criterion}: {crit_value:.4f}")
                  
                  if crit_value < self.best_criterion_value:
                      self.best_criterion_value = crit_value
                      self.best_model = {
                          'beta': beta,
                          'residuals': residuals,
                          'fitted': self.create_lag_matrix(data_array, p, q, residuals)[0] @ beta,
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
              diff_order = self.diff_orders[col]
              fitted = self.best_model['fitted'][:, i]
              max_lag = max(self.best_p, self.best_q, 1)
              
              # Adjust for lags and differencing when selecting initial values
              start_idx = diff_order * max_lag
              if start_idx == 0:
                  start_idx = max_lag  # At least account for the lag structure
              
              if len(self.original_data[col]) >= start_idx + len(fitted):
                  initial_values = [self.original_data[col].values[-start_idx-len(fitted)]]
              elif len(self.original_data[col]) > 0:
                  initial_values = [self.original_data[col].values[0]]
              else:
                  initial_values = [0]
              
              # Inverse difference the fitted values
              undiff_fitted = self.inverse_difference(self.original_data[col].values, fitted, diff_order, initial_values)
              
              # Compute the correct index for fitted values
              fitted_length = len(fitted)
              if fitted_length <= len(self.model_data):
                  fitted_index = self.model_data.index[-fitted_length:]
                  observed_data = self.original_data[col].values[-fitted_length:]
                  observed_index = self.original_data.index[-fitted_length:]
              else:
                  fitted_index = pd.date_range(
                      start=self.model_data.index[0],
                      periods=fitted_length,
                      freq=self.model_data.index.freq or 'M'
                  )
                  observed_data = self.original_data[col].values[-fitted_length:] if fitted_length <= len(self.original_data) else self.original_data[col].values
                  observed_index = self.original_data.index[-fitted_length:] if fitted_length <= len(self.original_data) else self.original_data.index
              
              plt.figure(figsize=(10, 4))
              plt.plot(observed_index, observed_data, label='Observed', alpha=0.7)
              plt.plot(fitted_index, undiff_fitted, label='Fitted', linestyle='--')
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
        
        # Generate forecast dates
        last_date = self.model_data.index[-1]
        freq = self.model_data.index.freq
        if freq is None:
            try:
                freq = pd.infer_freq(self.model_data.index)
            except:
                freq = 'M'  # Default to monthly if inference fails
        
        try:
            forecast_dates = pd.date_range(
                start=last_date + to_offset(freq), 
                periods=h, 
                freq=freq
            )
        except:
            # Fallback for irregular or failed offset
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=30),  # Approximate monthly offset
                periods=h, 
                freq='M'
            )
        
        forecast_df = pd.DataFrame(forecasts['point'], index=forecast_dates, columns=self.columns)
        for col in self.columns:
            col_idx = self.columns.index(col)
            forecast_df[f'{col}_ci_lower'] = forecasts['ci_lower'][:, col_idx]
            forecast_df[f'{col}_ci_upper'] = forecasts['ci_upper'][:, col_idx]
        
        return forecast_df