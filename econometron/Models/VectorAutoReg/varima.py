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


def process_time_series(data, date_column=None, columns=None, max_diff=2, significance_level=0.05, 
                       plot=True, log_transform=False):
    """
    Process time series data: handle missing values, check stationarity, and visualize ACF/PACF.
    
    Parameters:
    - data: DataFrame or Series with time series data
    - date_column: str, name of the date column (if DataFrame)
    - columns: list, columns to analyze (if None, all numeric columns are used)
    - max_diff: int, maximum differencing order
    - significance_level: float, p-value threshold for ADF test
    - plot: bool, whether to plot ACF/PACF
    - log_transform: bool or list, whether to apply log transformation (True for all, list for specific columns)
    
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
    
    # Determine which columns to log transform
    if log_transform is True:
        log_columns = columns
    elif isinstance(log_transform, list):
        log_columns = [col for col in log_transform if col in columns]
    else:
        log_columns = []
    
    # Validate log transformation feasibility
    for col in log_columns:
        if (data[col] <= 0).any():
            print(f"Warning: Column '{col}' contains non-positive values. Cannot apply log transformation.")
            log_columns = [c for c in log_columns if c != col]
    
    # Apply log transformation
    data_transformed = data.copy()
    for col in log_columns:
        data_transformed[col] = np.log(data[col])
        print(f"Applied log transformation to column: {col}")
    
    # Infer frequency if not set
    if data_transformed.index.inferred_freq is None:
        try:
            data_transformed.index.freq = pd.infer_freq(data_transformed.index)
        except:
            print("Could not infer frequency. Trying common frequencies...")
            common_freqs = ['M', 'Q', 'A', 'D']
            for freq in common_freqs:
                try:
                    data_transformed.index = pd.date_range(start=data_transformed.index[0], periods=len(data_transformed), freq=freq)
                    data_transformed.index.freq = freq
                    break
                except:
                    continue
            if data_transformed.index.inferred_freq is None:
                print("Setting to monthly frequency as fallback.")
                data_transformed.index = pd.date_range(start=data_transformed.index[0], periods=len(data_transformed), freq='M')
    
    results = {}
    
    for col in columns:
        print(f"\nProcessing column: {col}")
        series = data_transformed[col].copy()
        
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
        results[col] = {
            'original': series, 
            'log_transformed': col in log_columns,
            'adf_results': {}, 
            'stationary': False
        }
        
        # ADF test on original/transformed series
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
            plt.title(f'ACF - {col}' + (' (log-transformed)' if col in log_columns else ''))
            
            plt.subplot(122)
            plot_pacf(results[col].get('differenced', series), lags=10, ax=plt.gca())
            plt.title(f'PACF - {col}' + (' (log-transformed)' if col in log_columns else ''))
            
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
        """
        try:
            residuals = self.compute_residuals(data, params, p, q, K)
            if residuals.shape[0] <= K:
                return 1e10
            resid_cov = np.cov(residuals.T) + 1e-2 * np.eye(K)  # Stronger regularization
            min_eig = np.min(np.real(np.linalg.eigvals(resid_cov)))
            if min_eig <= 1e-8:
                resid_cov += (abs(min_eig) + 1e-2) * np.eye(K)
            try:
                log_det = np.log(np.linalg.det(resid_cov))
            except (np.linalg.LinAlgError, ValueError, OverflowError):
                return 1e10
            if not np.isfinite(log_det):
                return 1e10
            T = residuals.shape[0]
            ll = -0.5 * T * (K * np.log(2 * np.pi) + log_det)
            resid_cov_inv = np.linalg.pinv(resid_cov)
            for t in range(T):
                ll -= 0.5 * residuals[t] @ resid_cov_inv @ residuals[t]
            return -ll
        except Exception:
            return 1e10
    
    def compute_aic_bic(self, residuals, K, p, q, T):
        """
        Compute AIC and BIC for model evaluation.
        """
        try:
            resid_cov = np.cov(residuals.T) + 1e-2 * np.eye(K)
            min_eig = np.min(np.real(np.linalg.eigvals(resid_cov)))
            if min_eig <= 1e-8:
                resid_cov += (abs(min_eig) + 1e-2) * np.eye(K)
            try:
                log_det = np.log(np.linalg.det(resid_cov))
            except (np.linalg.LinAlgError, ValueError, OverflowError):
                return np.inf, np.inf
            if not np.isfinite(log_det):
                return np.inf, np.inf
            n_params = K * (1 + K * p + K * q)
            ll = -0.5 * T * (K * np.log(2 * np.pi) + log_det)
            for t in range(T):
                ll -= 0.5 * residuals[t] @ np.linalg.pinv(resid_cov) @ residuals[t]
            aic = -2 * ll + 2 * n_params
            bic = -2 * ll + n_params * np.log(T)
            return aic, bic
        except Exception:
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
              
              # Improved parameter initialization and bounds
              np.random.seed(42)
              initial_params = np.random.uniform(-0.1, 0.1, n_params)
              bounds = [(-0.99, 0.99)] * n_params
              
              # Optimize parameters
              result = minimize(
                  self.log_likelihood,
                  initial_params,
                  args=(data_array, p, q, K),
                  method='L-BFGS-B',
                  bounds=bounds,
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
    def plot_residuals(self):
        """
        Plot residual diagnostics including time series, ACF, histogram, and Q-Q plot.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting residuals.")
        
        residuals = self.best_model['residuals']
        n_vars = len(self.columns)
        
        fig, axes = plt.subplots(n_vars, 4, figsize=(16, 4 * n_vars))
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(self.columns):
            resid = residuals[:, i]
            
            # Time series plot
            axes[i, 0].plot(resid)
            axes[i, 0].set_title(f'Residuals - {col}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Residual')
            axes[i, 0].grid(True, alpha=0.3)
            
            # ACF plot
            max_lags = min(20, len(resid) // 4)
            try:
                acf_vals = acf(resid, nlags=max_lags, fft=False)
                axes[i, 1].stem(range(len(acf_vals)), acf_vals)
                axes[i, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                axes[i, 1].axhline(y=1.96/np.sqrt(len(resid)), color='r', linestyle='--', alpha=0.5)
                axes[i, 1].axhline(y=-1.96/np.sqrt(len(resid)), color='r', linestyle='--', alpha=0.5)
                axes[i, 1].set_title(f'Residual ACF - {col}')
                axes[i, 1].set_xlabel('Lag')
                axes[i, 1].set_ylabel('ACF')
                axes[i, 1].grid(True, alpha=0.3)
            except:
                axes[i, 1].text(0.5, 0.5, 'ACF computation failed', ha='center', va='center', transform=axes[i, 1].transAxes)
            
            # Histogram
            axes[i, 2].hist(resid, bins=20, density=True, alpha=0.7, edgecolor='black')
            # Overlay normal distribution
            x = np.linspace(resid.min(), resid.max(), 100)
            axes[i, 2].plot(x, (1/np.sqrt(2*np.pi*np.var(resid))) * np.exp(-0.5 * (x - np.mean(resid))**2 / np.var(resid)), 
                        'r-', label='Normal')
            axes[i, 2].set_title(f'Residual Distribution - {col}')
            axes[i, 2].set_xlabel('Residual')
            axes[i, 2].set_ylabel('Density')
            axes[i, 2].legend()
            axes[i, 2].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(resid, dist="norm", plot=axes[i, 3])
            axes[i, 3].set_title(f'Q-Q Plot - {col}')
            axes[i, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print residual statistics
        print("\nResidual Diagnostics Summary:")
        print("=" * 50)
        for i, col in enumerate(self.columns):
            resid = residuals[:, i]
            diag = self.residual_diag_results[col]
            
            print(f"\n{col}:")
            print(f"  Mean: {np.mean(resid):.6f}")
            print(f"  Std Dev: {np.std(resid):.6f}")
            print(f"  Skewness: {stats.skew(resid):.4f}")
            print(f"  Kurtosis: {stats.kurtosis(resid):.4f}")
            print(f"  Ljung-Box Test: Statistic = {diag['ljung_box']['statistic']:.4f}, p-value = {diag['ljung_box']['p_value']:.4f}")
            print(f"  Shapiro-Wilk Test: Statistic = {diag['shapiro_wilk']['statistic']:.4f}, p-value = {diag['shapiro_wilk']['p_value']:.4f}")
            
            # Interpretation
            if diag['ljung_box']['p_value'] > 0.05:
                print(f"  ✓ No significant autocorrelation detected")
            else:
                print(f"  ✗ Significant autocorrelation detected")
                
            if diag['shapiro_wilk']['p_value'] > 0.05:
                print(f"  ✓ Residuals appear normally distributed")
            else:
                print(f"  ✗ Residuals may not be normally distributed")

    def plot_forecast(self, h=None, include_history=True, history_periods=50):
        """
        Plot forecasts with confidence intervals.
        
        Parameters:
        - h: Forecast horizon (defaults to self.forecast_horizon)
        - include_history: bool, whether to include historical data
        - history_periods: int, number of historical periods to show
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting forecasts.")
        
        h = h or self.forecast_horizon
        forecast_df = self.predict(h)
        
        n_vars = len(self.columns)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars))
        if n_vars == 1:
            axes = [axes]
        
        for i, col in enumerate(self.columns):
            ax = axes[i]
            
            # Historical data
            if include_history:
                hist_data = self.original_data[col].iloc[-history_periods:]
                ax.plot(hist_data.index, hist_data.values, 'b-', label='Historical', linewidth=2)
            
            # Forecasts
            forecast_values = forecast_df[col]
            ci_lower = forecast_df[f'{col}_ci_lower']
            ci_upper = forecast_df[f'{col}_ci_upper']
            
            ax.plot(forecast_df.index, forecast_values, 'r-', label='Forecast', linewidth=2)
            ax.fill_between(forecast_df.index, ci_lower, ci_upper, alpha=0.3, color='red', label='95% CI')
            
            # Add vertical line at forecast start
            if include_history:
                ax.axvline(x=self.original_data.index[-1], color='black', linestyle='--', alpha=0.7, label='Forecast Start')
            
            ax.set_title(f'Forecast for {col}' + (' (log scale)' if self.stationarity_results[col].get('log_transformed', False) else ''))
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add forecast statistics
            mean_forecast = np.mean(forecast_values)
            std_forecast = np.std(forecast_values)
            ax.text(0.02, 0.98, f'Mean: {mean_forecast:.4f}\nStd: {std_forecast:.4f}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print forecast summary
        print("\nForecast Summary:")
        print("=" * 50)
        for col in self.columns:
            forecast_values = forecast_df[col]
            ci_lower = forecast_df[f'{col}_ci_lower']
            ci_upper = forecast_df[f'{col}_ci_upper']
            
            print(f"\n{col}:")
            print(f"  Mean Forecast: {np.mean(forecast_values):.4f}")
            print(f"  Forecast Range: [{np.min(forecast_values):.4f}, {np.max(forecast_values):.4f}]")
            print(f"  Average CI Width: {np.mean(ci_upper - ci_lower):.4f}")
            if self.stationarity_results[col].get('log_transformed', False):
                print(f"  Note: Values are in log scale")

    def plot_diagnostics(self):
        """
        Comprehensive diagnostic plots including model fit, residuals, and stability checks.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting diagnostics.")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        n_vars = len(self.columns)
        
        # Plot 1: Model Selection Criteria
        if hasattr(self, 'all_results') and self.all_results:
            ax1 = plt.subplot(3, 3, 1)
            p_values = [result['p'] for result in self.all_results]
            q_values = [result['q'] for result in self.all_results]
            aic_values = [result['aic'] for result in self.all_results]
            bic_values = [result['bic'] for result in self.all_results]
            
            # Create a scatter plot of AIC/BIC values
            criterion_values = aic_values if self.criterion == 'AIC' else bic_values
            scatter = ax1.scatter(p_values, q_values, c=criterion_values, cmap='viridis', s=100)
            ax1.scatter(self.best_p, self.best_q, c='red', s=200, marker='*', label='Best Model')
            ax1.set_xlabel('AR Order (p)')
            ax1.set_ylabel('MA Order (q)')
            ax1.set_title(f'Model Selection ({self.criterion})')
            ax1.legend()
            plt.colorbar(scatter, ax=ax1, label=self.criterion)
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coefficient Significance
        ax2 = plt.subplot(3, 3, 2)
        coef_data = []
        coef_labels = []
        for idx, row in self.coefficient_table.iterrows():
            for col in self.columns:
                if f'{col}_coef' in row:
                    coef_data.append(row[f'{col}_coef'])
                    coef_labels.append(f'{idx}_{col}')
        
        colors = ['red' if abs(coef) < 1.96 * 0.01 else 'blue' for coef in coef_data]  # Rough significance check
        ax2.barh(range(len(coef_data)), coef_data, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(coef_data)))
        ax2.set_yticklabels(coef_labels, fontsize=8)
        ax2.set_xlabel('Coefficient Value')
        ax2.set_title('Model Coefficients')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residual Variance over Time
        ax3 = plt.subplot(3, 3, 3)
        residuals = self.best_model['residuals']
        window_size = max(5, len(residuals) // 10)
        for i, col in enumerate(self.columns):
            resid = residuals[:, i]
            rolling_var = pd.Series(resid).rolling(window=window_size).var()
            ax3.plot(rolling_var, label=col, alpha=0.7)
        ax3.set_title('Rolling Residual Variance')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plots 4-6: Individual variable diagnostics
        for i, col in enumerate(self.columns[:3]):  # Limit to first 3 variables for space
            ax = plt.subplot(3, 3, 4 + i)
            
            # Fitted vs Actual
            fitted = self.best_model['fitted'][:, i]
            actual = self.model_data[col].values[-len(fitted):]
            
            ax.scatter(actual, fitted, alpha=0.6)
            min_val, max_val = min(np.min(actual), np.min(fitted)), max(np.max(actual), np.max(fitted))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Fitted')
            ax.set_title(f'Fitted vs Actual - {col}')
            ax.grid(True, alpha=0.3)
            
            # Calculate R-squared
            ss_res = np.sum((actual - fitted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plots 7-9: Residual Analysis by Variable
        for i, col in enumerate(self.columns[:3]):
            ax = plt.subplot(3, 3, 7 + i)
            resid = residuals[:, i]
            
            # Standardized residuals
            std_resid = resid / np.std(resid)
            ax.plot(std_resid, 'o-', alpha=0.6, markersize=3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='±2σ')
            ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'Standardized Residuals - {col}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Std. Residual')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Count outliers
            outliers = np.sum(np.abs(std_resid) > 2)
            ax.text(0.05, 0.95, f'Outliers: {outliers}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print comprehensive diagnostic summary
        print("\nComprehensive Model Diagnostics:")
        print("=" * 60)
        print(f"Best Model: VARIMA({self.best_p}, d, {self.best_q})")
        print(f"Model Selection Criterion ({self.criterion}): {self.best_criterion_value:.4f}")
        print(f"Number of Parameters: {self.coefficient_table.shape[0] * len(self.columns)}")
        print(f"Sample Size: {self.model_data.shape[0]}")
        
        # Model fit statistics
        print(f"\nModel Fit Statistics:")
        for i, col in enumerate(self.columns):
            fitted = self.best_model['fitted'][:, i]
            actual = self.model_data[col].values[-len(fitted):]
            
            # Calculate various fit metrics
            mse = np.mean((actual - fitted) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual - fitted))
            mape = np.mean(np.abs((actual - fitted) / actual)) * 100 if np.all(actual != 0) else np.inf
            
            ss_res = np.sum((actual - fitted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"  {col}:")
            print(f"    R-squared: {r_squared:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    MAPE: {mape:.2f}%")
        
        # Residual diagnostics summary
        print(f"\nResidual Diagnostics Summary:")
        all_ljung_box_ok = True
        all_normality_ok = True
        
        for col in self.columns:
            diag = self.residual_diag_results[col]
            ljung_box_ok = diag['ljung_box']['p_value'] > 0.05
            normality_ok = diag['shapiro_wilk']['p_value'] > 0.05
            
            if not ljung_box_ok:
                all_ljung_box_ok = False
            if not normality_ok:
                all_normality_ok = False
        
        print(f"  Ljung-Box Test (No Autocorrelation): {'✓ PASS' if all_ljung_box_ok else '✗ FAIL'}")
        print(f"  Shapiro-Wilk Test (Normality): {'✓ PASS' if all_normality_ok else '✗ FAIL'}")
        
        # Model stability check
        print(f"\nModel Stability:")
        if hasattr(self, 'best_model') and 'beta' in self.best_model:
            max_coef = np.max(np.abs(self.best_model['beta']))
            print(f"  Maximum coefficient magnitude: {max_coef:.4f}")
            if max_coef < 1.0:
                print(f"  ✓ Model appears stable (all coefficients < 1)")
            else:
                print(f"  ⚠ Model may be unstable (coefficients ≥ 1)")