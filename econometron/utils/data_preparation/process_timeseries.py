import pandas as pd
import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional

class TransformTS:
    """
    A class for transforming time series data with stationarity checks and analysis.

    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Input time series data.
    columns : list or None, optional
        List of columns to transform. If None, all numeric columns are selected.
    method : str, optional
        Transformation method: 'diff', 'boxcox', 'log', 'log-diff', 'hp', 'inverse'.
        Default is 'diff'.
    demean : bool, optional
        If True, demean the data before transformation. Default is True.
    analysis : bool, optional
        If True, perform time series analysis (summary, correlation, ACF/PACF). Default is True.
    plot : bool, optional
        If True, generate diagnostic plots (time series, ACF, PACF). Default is False.
    lamb : float, optional
        Lambda parameter for Hodrick-Prescott filter. Default is 1600.
    log_data : bool, optional
        If True, apply log transformation for 'log' or 'log-diff' methods when data is not in log form.
        Default is True.
    max_diff : int, optional
        Maximum differencing order before switching to log-diff for non-stationary series.
        Default is 2.
    """

    def __init__(self, data: Union[pd.DataFrame, pd.Series], columns: Optional[List[str]] = None, 
                 method: str = 'diff', demean: bool = True, analysis: bool = True, 
                 plot: bool = False, lamb: float = 1600, log_data: bool = True, max_diff: int = 2):
        self.data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data.copy())
        self.columns = columns if columns else self.data.select_dtypes(include=np.number).columns.tolist()
        self.method = method.lower()
        self.demean = demean
        self.analysis = analysis
        self.plot = plot
        self.lamb = lamb
        self.log_data = log_data
        self.max_diff = max_diff
        self.transformed_data = None
        self.original_data = self.data[self.columns].copy()
        self.lambda_boxcox = {}  # Store Box-Cox lambda parameters for inverse transform
        self.stationary_status = {}  # Store stationarity results
        self.is_log = {}  # Track if data is already in log form
        self.diff_order = {}  # Track differencing order per column

        # Validate inputs
        self._validate_inputs()

        # Check stationarity (for reporting or diff method)
        self._check_stationarity_all()

        # Apply transformations
        self.transform()

        # Perform analysis if requested
        if self.analysis:
            self.analyze()

    def _validate_inputs(self):
        """Validate input data and parameters."""
        if not self.columns:
            raise ValueError("No numeric columns found in the data.")
        
        if not all(col in self.data.columns for col in self.columns):
            raise ValueError("Specified columns not found in the data.")
            
        valid_methods = ['diff', 'boxcox', 'log', 'log-diff', 'hp', 'inverse']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}.")
            
        if self.data[self.columns].isna().any().any():
            print("Warning: NaN values detected. Consider handling them before transformation.")
        
        if self.max_diff < 1:
            raise ValueError("max_diff must be at least 1.")

    def _check_stationarity(self, series: pd.Series, col: str) -> bool:
        """Perform ADF test to check if a series is stationary."""
        result = adfuller(series.dropna(), autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05  # 5% significance level
        self.stationary_status[col] = {
            'is_stationary': is_stationary,
            'p_value': p_value,
            'adf_statistic': result[0]
        }
        return is_stationary

    def _check_stationarity_all(self):
        """Check stationarity of all columns for reporting."""
        for col in self.columns:
            self._check_stationarity(self.data[col], col)

    def _check_if_log(self, series: pd.Series) -> bool:
        """Check if a series is likely in log form based on its properties."""
        if (series <= 0).any():
            return False  # Log-transformed data should be positive
        # Heuristic: check if the range is consistent with log-transformed data
        if series.max() - series.min() < 10:  # Arbitrary threshold for log-like behavior
            return True
        return False

    def _make_stationary(self, series: pd.Series, col: str) -> pd.Series:
        """Apply differencing until stationary or switch to log-diff if over-differencing."""
        diff_count = 0
        current_series = series.copy()
        
        while not self._check_stationarity(current_series, col) and diff_count < self.max_diff:
            current_series = current_series.diff().dropna()
            diff_count += 1
            self.diff_order[col] = diff_count
        
        if diff_count >= self.max_diff and not self.stationary_status[col]['is_stationary']:
            print(f"Column {col} requires excessive differencing. Switching to log-diff.")
            current_series = series.copy()
            if self.log_data:
                if (current_series <= 0).any():
                    print(f"Warning: Column {col} contains non-positive values. Setting inf to NaN and dropping NaNs.")
                    current_series = np.log(current_series.replace(0, np.nan))
                    current_series.replace([np.inf, -np.inf], np.nan, inplace=True)
                    current_series = current_series.dropna()
                else:
                    current_series = np.log(current_series)
                self.is_log[col] = True
            current_series = current_series.diff().dropna()
            self.diff_order[col] = 1  # Log-diff counts as one difference
        
        return current_series

    def transform(self):
        """Apply the specified transformation to the selected columns."""
        self.transformed_data = self.data[self.columns].copy()
        
        for col in self.columns:
            series = self.transformed_data[col].copy()
            
            # Demean if requested
            if self.demean:
                series = series - series.mean()
                
            # Check if data is in log form for log-related methods
            self.is_log[col] = self._check_if_log(series) if self.method in ['log', 'log-diff'] else False
            
            if self.method == 'diff':
                # Apply differencing until stationary
                self.transformed_data[col] = self._make_stationary(series, col)
                
            elif self.method == 'boxcox':
                if (series <= 0).any():
                    raise ValueError(f"Column {col} contains non-positive values, cannot apply Box-Cox.")
                self.transformed_data[col], self.lambda_boxcox[col] = boxcox(series)
                
            elif self.method == 'log':
                if self.is_log[col] and self.log_data:
                    print(f"Column {col} appears to be in log form, skipping log transformation.")
                elif (series <= 0).any():
                    print(f"Warning: Column {col} contains non-positive values. Setting inf to NaN and dropping NaNs.")
                    series = np.log(series.replace(0, np.nan))
                    series.replace([np.inf, -np.inf], np.nan, inplace=True)
                    series.dropna(inplace=True)
                    self.transformed_data[col] = series
                    self.is_log[col] = True
                else:
                    self.transformed_data[col] = np.log(series)
                    self.is_log[col] = True
                
            elif self.method == 'log-diff':
                if self.is_log[col] and self.log_data:
                    print(f"Column {col} appears to be in log form, applying differencing only.")
                    self.transformed_data[col] = series.diff().dropna()
                    self.diff_order[col] = 1
                elif (series <= 0).any() and self.log_data:
                    print(f"Warning: Column {col} contains non-positive values. Setting inf to NaN and dropping NaNs.")
                    series = np.log(series.replace(0, np.nan))
                    series.replace([np.inf, -np.inf], np.nan, inplace=True)
                    series = series.diff().dropna()
                    self.transformed_data[col] = series
                    self.is_log[col] = True
                    self.diff_order[col] = 1
                else:
                    series = np.log(series) if self.log_data else series
                    self.transformed_data[col] = series.diff().dropna()
                    self.is_log[col] = self.log_data
                    self.diff_order[col] = 1
                
            elif self.method == 'hp':
                cycle, trend = hpfilter(series.dropna(), lamb=self.lamb)
                self.transformed_data[col] = cycle
                
            elif self.method == 'inverse':
                self.transformed_data[col] = self._inverse_transform(series, col)
                
        return self.transformed_data
    
    def _inverse_transform(self, series: pd.Series, col: str) -> pd.Series:
        """Apply inverse transformation based on the method."""
        if self.method == 'diff':
            result = series.copy()
            for _ in range(self.diff_order.get(col, 1)):
                result = result.cumsum()
            return result
        elif self.method == 'boxcox':
            if col not in self.lambda_boxcox:
                raise ValueError(f"No Box-Cox lambda found for column {col}.")
            lamb = self.lambda_boxcox[col]
            if lamb == 0:
                return np.exp(series)
            else:
                return np.power((series * lamb + 1), 1 / lamb)
        elif self.method == 'log':
            return np.exp(series)
        elif self.method == 'log-diff':
            result = series.cumsum()
            return np.exp(result) if self.is_log.get(col, False) else result
        else:
            raise ValueError("Inverse transform not applicable for this method.")
            
    def analyze(self):
        """Perform time series analysis."""
        print("\n=== Time Series Analysis ===")
        
        # Stationarity results
        print("\nStationarity Check (ADF Test):")
        for col, status in self.stationary_status.items():
            print(f"Column {col}: {'Stationary' if status['is_stationary'] else 'Non-stationary'}, "
                  f"p-value: {status['p_value']:.4f}, ADF Statistic: {status['adf_statistic']:.4f}")
        
        # Summary statistics
        print("\nSummary Statistics:")
        print(self.transformed_data.describe())
        
        # NaN counts
        nan_counts = self.transformed_data.isna().sum()
        print("\nNaN Counts:")
        print(nan_counts)
        
        # Correlation matrix
        if len(self.columns) > 1:
            print("\nCorrelation Matrix:")
            print(self.transformed_data.corr())
        
        # Plotting
        if self.plot:
            for col in self.columns:
                plt.figure(figsize=(12, 4))
                
                # Time series plot
                plt.subplot(1, 3, 1)
                plt.plot(self.transformed_data[col], label=f'Transformed {col}')
                plt.title(f'Transformed Series: {col}')
                plt.legend()
                
                # ACF plot
                plt.subplot(1, 3, 2)
                acf_vals = acf(self.transformed_data[col].dropna(), nlags=20)
                plt.stem(range(len(acf_vals)), acf_vals)
                plt.title(f'ACF: {col}')
                
                # PACF plot
                plt.subplot(1, 3, 3)
                pacf_vals = pacf(self.transformed_data[col].dropna(), nlags=20)
                plt.stem(range(len(pacf_vals)), pacf_vals)
                plt.title(f'PACF: {col}')
                
                plt.tight_layout()
                plt.show()
                
    def get_transformed_data(self) -> pd.DataFrame:
        """Return the transformed data."""
        return self.transformed_data.dropna()

    def trns_info(self) -> dict:
        """
        Retrieve transformation and stationarity information for each column.
        
        Returns:
        --------
        dict
            A dictionary containing transformation details, differencing order, 
            and stationarity information for each column.
        """
        info = {}
        
        for col in self.columns:
            info[col] = {
                'transformation_method': self.method,
                'differencing_order': self.diff_order.get(col, 0),
                'is_stationary': self.stationary_status.get(col, {}).get('is_stationary', False),
                'p_value': self.stationary_status.get(col, {}).get('p_value', None),
                'adf_statistic': self.stationary_status.get(col, {}).get('adf_statistic', None),
                'is_log_transformed': self.is_log.get(col, False),
                'boxcox_lambda': self.lambda_boxcox.get(col, None),
                'original_stationarity': self.stationary_status.get(col, {}).get('is_stationary', False)
            }
            
            # Additional details based on transformation method
            if self.method == 'diff':
                if self.diff_order.get(col, 0) == 0:
                    info[col]['details'] = "No differencing applied (series was already stationary)."
                else:
                    info[col]['details'] = (f"Applied differencing {info[col]['differencing_order']} time(s) "
                                          f"to achieve stationarity.")
            elif self.method == 'log-diff':
                info[col]['details'] = (f"Applied {'log transformation and ' if self.log_data and not self.is_log[col] else ''}"
                                      f"differencing (order {info[col]['differencing_order']}).")
            elif self.method == 'boxcox':
                info[col]['details'] = (f"Applied Box-Cox transformation with lambda = {info[col]['boxcox_lambda']:.4f}.")
            elif self.method == 'log':
                info[col]['details'] = (f"Applied log transformation{' (skipped as data was in log form)' if self.is_log[col] else ''}.")
            elif self.method == 'hp':
                info[col]['details'] = f"Applied Hodrick-Prescott filter with lambda = {self.lamb}."
            elif self.method == 'inverse':
                info[col]['details'] = f"Applied inverse transformation for {self.method}."
        
        return info