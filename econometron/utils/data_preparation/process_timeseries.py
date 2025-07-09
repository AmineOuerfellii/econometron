import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import boxcox
import matplotlib.pyplot as plt

class TimeSeriesProcessor:

    def __init__(self, data, columns=None, method='diff', analysis=True, plot=False):
        """
        Initialize TimeSeriesProcessor.
        
        Parameters:
        - data: pandas DataFrame or numpy array
        - columns: list of column names to process (if None, all columns are processed)
        - method: transformation method ('diff', 'log_diff', 'detrend', 'seasonal', 'boxcox', 'hodrick_prescott')
        - analysis: bool, whether to perform stationarity analysis
        - plot: bool, whether to generate ACF/PACF plots
        """
        self.data = pd.DataFrame(data, columns=columns) if isinstance(data, np.ndarray) else data
        self.columns = columns if columns is not None else self.data.columns
        self.method = method
        self.analysis = analysis
        self.plot = plot
        self.stationary_info = {}
        self.transformed_data = None
        self.original_data = self.data.copy()
        self.transform_params = {}  # Store parameters for inverse transformation
        
        if self.analysis:
            self._analyze_stationarity()

    def _check_stationarity(self, series, col_name):
        """Perform ADF test to check stationarity."""
        result = adfuller(series.dropna(), autolag='AIC')
        is_stationary = result[1] < 0.05  # p-value < 0.05 indicates stationarity
        self.stationary_info[col_name] = {
            'is_stationary': is_stationary,
            'p_value': result[1],
            'adf_statistic': result[0],
            'transformation_applied': None,
            'order': 0
        }
        return is_stationary

    def _apply_transformation(self, series, col_name):
        """Apply specified transformation to make series stationary."""
        original_series = series.copy()
        order = 0
        max_diff = 6 # Maximum differencing order

        if self.method == 'diff':
            while not self._check_stationarity(series, col_name) and order < max_diff:
                series = series.diff().dropna()
                order += 1
            self.transform_params[col_name] = {'method': 'diff', 'order': order}
            
        elif self.method == 'log_diff':
            series = np.log(series + 1e-10)  # Avoid log(0)
            while not self._check_stationarity(series, col_name) and order < max_diff:
                series = series.diff().dropna()
                order += 1
            self.transform_params[col_name] = {'method': 'log_diff', 'order': order}
            
        elif self.method == 'detrend':
            series = series - series.rolling(window=12).mean()
            self.transform_params[col_name] = {'method': 'detrend'}
            
        elif self.method == 'seasonal':
            decomposition = seasonal_decompose(series, period=12, model='additive', extrapolate_trend='freq')
            series = decomposition.resid
            self.transform_params[col_name] = {'method': 'seasonal', 'trend': decomposition.trend, 'seasonal': decomposition.seasonal}
            
        elif self.method == 'boxcox':
            series, lmbda = boxcox(series + 1e-10)  # Avoid zero
            self.transform_params[col_name] = {'method': 'boxcox', 'lambda': lmbda}
            
        elif self.method == 'hodrick_prescott':
            cycle, trend = hpfilter(series, lamb=1600)
            series = cycle
            self.transform_params[col_name] = {'method': 'hodrick_prescott', 'trend': trend}
            
        self.stationary_info[col_name]['transformation_applied'] = self.method
        self.stationary_info[col_name]['order'] = order
        return series

    def _analyze_stationarity(self):
        """Analyze stationarity for each column and apply transformations if needed."""
        self.transformed_data = self.data.copy()
        for col in self.columns:
            series = self.data[col].copy()
            if not self._check_stationarity(series, col):
                self.transformed_data[col] = self._apply_transformation(series, col)
            
            if self.plot:
                self._plot_acf_pacf(self.transformed_data[col], col)

    def _plot_acf_pacf(self, series, col_name):
        """Plot ACF and PACF for a given series."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(121)
        acf_vals = acf(series.dropna(), nlags=20)
        plt.stem(acf_vals)
        plt.title(f'ACF - {col_name}')
        
        plt.subplot(122)
        pacf_vals = pacf(series.dropna(), nlags=20)
        plt.stem(pacf_vals)
        plt.title(f'PACF - {col_name}')
        
        plt.tight_layout()
        plt.show()

    def get_stationarity_info(self):
        """Return stationarity analysis results."""
        return self.stationary_info

    def get_transformed_data(self):
        """Return transformed data."""
        return self.transformed_data.dropna()

    def untransform(self, data=None, column=None, method=None, **kwargs):
        """
        Inverse transform the data.
        
        Parameters:
        - data: DataFrame or Series to untransform (if None, uses transformed_data)
        - column: specific column to untransform (if None, untransform all)
        - method: transformation method (if None, uses stored method)
        - **kwargs: additional parameters (e.g., order, lambda, trend)
        """
        if data is None:
            data = self.transformed_data.copy()
        else:
            data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data.copy()

        if column:
            columns = [column]
        else:
            columns = self.columns

        result = data.copy()
        for col in columns:
            if col not in self.transform_params:
                continue
                
            params = self.transform_params.get(col, {})
            method = method or params.get('method')
            
            if method == 'diff':
                order = kwargs.get('order', params.get('order', 1))
                result[col] = self._inverse_diff(data[col], order)
                
            elif method == 'log_diff':
                order = kwargs.get('order', params.get('order', 1))
                series = self._inverse_diff(data[col], order)
                result[col] = np.exp(series)
                
            elif method == 'detrend':
                result[col] = data[col] + self.original_data[col].rolling(window=12).mean()
                
            elif method == 'seasonal':
                trend = kwargs.get('trend', params.get('trend'))
                seasonal = kwargs.get('seasonal', params.get('seasonal'))
                if trend is not None and seasonal is not None:
                    result[col] = data[col] + trend + seasonal
                    
            elif method == 'boxcox':
                lmbda = kwargs.get('lambda', params.get('lambda'))
                if lmbda is not None:
                    result[col] = self._inverse_boxcox(data[col], lmbda)
                    
            elif method == 'hodrick_prescott':
                trend = kwargs.get('trend', params.get('trend'))
                if trend is not None:
                    result[col] = data[col] + trend
                    
        return result

    def _inverse_diff(self, series, order):
        """Inverse differencing transformation."""
        result = series.copy()
        for _ in range(order):
            result = result.cumsum() + self.original_data[series.name].shift(1).fillna(0)
        return result

    def _inverse_boxcox(self, series, lmbda):
        """Inverse Box-Cox transformation."""
        if lmbda == 0:
            return np.exp(series)
        else:
            return (series * lmbda + 1) ** (1 / lmbda)