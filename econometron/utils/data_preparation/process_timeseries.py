import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
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
        self.original_means = {}  # Store original means for inverse transformation

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
        clean_series = series.dropna()
        if len(clean_series) < 12:  # Minimum observations for ADF test
            print(f"Warning: Column {col} has insufficient data for stationarity test.")
            return False
            
        result = adfuller(clean_series, autolag='AIC')
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

    def transform(self):
        """Apply the specified transformation to the selected columns."""
        self.transformed_data = self.data.copy()
        
        for col in self.columns:
            series = self.data[col].copy()
            
            # Store original mean for inverse transformation
            self.original_means[col] = series.mean()
            
            # Handle NaN values before transformation
            if series.isna().any():
                print(f"Warning: Column {col} contains NaN values. Dropping NaNs before transformation.")
                series = series.dropna()
            
            # Demean if requested (store mean before demeaning)
            if self.demean:
                series = series - series.mean()
                
            # Check if data is in log form for log-related methods
            self.is_log[col] = self._check_if_log(series) if self.method in ['log', 'log-diff'] else False
            
            if self.method == 'diff':
                # Apply differencing until stationary
                transformed_series = self._make_stationary(series, col)
                # Create a new series with proper index alignment
                self.transformed_data[col] = np.nan
                self.transformed_data.loc[transformed_series.index, col] = transformed_series
                
            elif self.method == 'boxcox':
                if (series <= 0).any():
                    raise ValueError(f"Column {col} contains non-positive values, cannot apply Box-Cox.")
                # Apply Box-Cox transformation
                transformed, self.lambda_boxcox[col] = boxcox(series)
                # Create a new series with proper index alignment
                self.transformed_data[col] = np.nan
                self.transformed_data.loc[series.index, col] = transformed
                
            elif self.method == 'log':
                if self.is_log[col] and self.log_data:
                    print(f"Column {col} appears to be in log form, skipping log transformation.")
                    self.transformed_data[col] = np.nan
                    self.transformed_data.loc[series.index, col] = series
                else:
                    if (series <= 0).any():
                        print(f"Warning: Column {col} contains non-positive values. Adding small constant and applying log.")
                        # Add small constant to make all values positive
                        min_val = series.min()
                        if min_val <= 0:
                            series = series - min_val + 1e-8
                    
                    transformed = np.log(series)
                    # Handle any remaining inf/-inf values
                    transformed = transformed.replace([np.inf, -np.inf], np.nan)
                    self.transformed_data[col] = np.nan
                    self.transformed_data.loc[transformed.index, col] = transformed
                    self.is_log[col] = True
                
            elif self.method == 'log-diff':
                if self.is_log[col] and self.log_data:
                    print(f"Column {col} appears to be in log form, applying differencing only.")
                    transformed = series.diff().dropna()
                    self.diff_order[col] = 1
                else:
                    if (series <= 0).any() and self.log_data:
                        print(f"Warning: Column {col} contains non-positive values. Adding small constant and applying log-diff.")
                        # Add small constant to make all values positive
                        min_val = series.min()
                        if min_val <= 0:
                            series = series - min_val + 1e-8
                    
                    if self.log_data:
                        transformed = np.log(series)
                        transformed = transformed.replace([np.inf, -np.inf], np.nan)
                        self.is_log[col] = True
                    else:
                        transformed = series
                        self.is_log[col] = False
                    
                    transformed = transformed.diff().dropna()
                    self.diff_order[col] = 1
                
                # Create a new series with proper index alignment
                self.transformed_data[col] = np.nan
                self.transformed_data.loc[transformed.index, col] = transformed
                
            elif self.method == 'hp':
                clean_series = series.dropna()
                cycle, trend = hpfilter(clean_series, lamb=self.lamb)
                # Create a new series with proper index alignment
                self.transformed_data[col] = np.nan
                self.transformed_data.loc[clean_series.index, col] = cycle
                
            elif self.method == 'inverse':
                # For inverse transformation, we need to know what the original transformation was
                # This assumes the data is already transformed and we want to reverse it
                raise NotImplementedError("Inverse transformation requires knowing the original transformation method and parameters.")
        
        return self.transformed_data

    def _make_stationary(self, series: pd.Series, col: str) -> pd.Series:
        """Apply differencing until stationary or switch to log-diff if over-differencing."""
        diff_count = 0
        current_series = series.copy()
        
        # First check if already stationary
        if self._check_stationarity(current_series, col):
            self.diff_order[col] = 0
            return current_series
        
        while not self._check_stationarity(current_series, col) and diff_count < self.max_diff:
            current_series = current_series.diff().dropna()
            diff_count += 1
            
            if len(current_series) < 12:  # Not enough data left
                print(f"Warning: Column {col} has insufficient data after {diff_count} differencing steps.")
                break
        
        self.diff_order[col] = diff_count
        
        if diff_count >= self.max_diff and not self.stationary_status[col]['is_stationary']:
            print(f"Column {col} requires excessive differencing. Switching to log-diff.")
            current_series = series.copy()
            
            if self.log_data:
                if (current_series <= 0).any():
                    print(f"Warning: Column {col} contains non-positive values. Adding small constant and applying log-diff.")
                    min_val = current_series.min()
                    if min_val <= 0:
                        current_series = current_series - min_val + 1e-8
                
                current_series = np.log(current_series)
                current_series = current_series.replace([np.inf, -np.inf], np.nan)
                self.is_log[col] = True
            
            current_series = current_series.diff().dropna()
            self.diff_order[col] = 1  # Log-diff counts as one difference
        
        return current_series

    def inverse_transform(self, col: str, transformed_series: pd.Series = None) -> pd.Series:
        """
        Apply inverse transformation for a specific column.
        
        Parameters:
        -----------
        col : str
            Column name to inverse transform
        transformed_series : pd.Series, optional
            Series to inverse transform. If None, uses the transformed data for this column.
            
        Returns:
        --------
        pd.Series
            Inverse transformed series
        """
        if transformed_series is None:
            if self.transformed_data is None:
                raise ValueError("No transformed data available. Run transform() first.")
            transformed_series = self.transformed_data[col].dropna()
        
        if self.method == 'diff':
            # Reverse differencing
            result = transformed_series.copy()
            diff_order = self.diff_order.get(col, 0)
            
            if diff_order == 0:
                return result
                
            # For proper inverse differencing, we need the original values
            # This is a simplified approach - in practice, you'd need the initial conditions
            for _ in range(diff_order):
                result = result.cumsum()
            
            # Add back the mean if it was removed
            if self.demean:
                result = result + self.original_means[col]
            
            return result
            
        elif self.method == 'boxcox':
            if col not in self.lambda_boxcox:
                raise ValueError(f"No Box-Cox lambda found for column {col}.")
            
            lamb = self.lambda_boxcox[col]
            result = inv_boxcox(transformed_series, lamb)
            
            # Add back the mean if it was removed
            if self.demean:
                result = result + self.original_means[col]
            
            return pd.Series(result, index=transformed_series.index)
            
        elif self.method == 'log':
            result = np.exp(transformed_series)
            
            # Add back the mean if it was removed (in original scale)
            if self.demean:
                # Note: This is approximate since mean was removed before log transformation
                result = result * np.exp(self.original_means[col])
            
            return result
            
        elif self.method == 'log-diff':
            # First reverse the differencing
            result = transformed_series.cumsum()
            
            # Then reverse log transformation if it was applied
            if self.is_log.get(col, False):
                result = np.exp(result)
            
            # Add back the mean if it was removed
            if self.demean:
                if self.is_log.get(col, False):
                    result = result * np.exp(self.original_means[col])
                else:
                    result = result + self.original_means[col]
            
            return result
            
        elif self.method == 'hp':
            # HP filter inverse is not straightforward as we only kept the cycle
            # We would need the trend component as well
            raise ValueError("Inverse transformation for HP filter requires the trend component.")
            
        else:
            raise ValueError(f"Inverse transform not implemented for method: {self.method}")
            
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
        print(self.transformed_data[self.columns].describe())
        
        # NaN counts
        nan_counts = self.transformed_data[self.columns].isna().sum()
        print("\nNaN Counts:")
        print(nan_counts)
        
        # Correlation matrix
        if len(self.columns) > 1:
            print("\nCorrelation Matrix:")
            corr_data = self.transformed_data[self.columns].dropna()
            if not corr_data.empty:
                print(corr_data.corr())
        
        # Plotting
        if self.plot:
            for col in self.columns:
                col_data = self.transformed_data[col].dropna()
                if len(col_data) < 10:
                    print(f"Skipping plots for {col}: insufficient data after transformation.")
                    continue
                    
                plt.figure(figsize=(12, 4))
                
                # Time series plot
                plt.subplot(1, 3, 1)
                plt.plot(col_data.index, col_data.values, label=f'Transformed {col}')
                plt.title(f'Transformed Series: {col}')
                plt.legend()
                
                # ACF plot
                plt.subplot(1, 3, 2)
                try:
                    acf_vals = acf(col_data, nlags=min(20, len(col_data)//4))
                    plt.stem(range(len(acf_vals)), acf_vals)
                    plt.title(f'ACF: {col}')
                except:
                    plt.text(0.5, 0.5, 'ACF calculation failed', ha='center', va='center')
                    plt.title(f'ACF: {col}')
                
                # PACF plot
                plt.subplot(1, 3, 3)
                try:
                    pacf_vals = pacf(col_data, nlags=min(20, len(col_data)//4))
                    plt.stem(range(len(pacf_vals)), pacf_vals)
                    plt.title(f'PACF: {col}')
                except:
                    plt.text(0.5, 0.5, 'PACF calculation failed', ha='center', va='center')
                    plt.title(f'PACF: {col}')
                
                plt.tight_layout()
                plt.show()
                
    def get_transformed_data(self) -> pd.DataFrame:
        """Return the transformed data."""
        return self.transformed_data[self.columns].dropna()

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
                'original_stationarity': self.stationary_status.get(col, {}).get('is_stationary', False),
                'demeaned': self.demean,
                'original_mean': self.original_means.get(col, None)
            }
            
            # Additional details based on transformation method
            if self.method == 'diff':
                if self.diff_order.get(col, 0) == 0:
                    info[col]['details'] = "No differencing applied (series was already stationary)."
                else:
                    info[col]['details'] = (f"Applied differencing {info[col]['differencing_order']} time(s) "
                                          f"to achieve stationarity.")
            elif self.method == 'log-diff':
                log_part = "log transformation and " if self.log_data and not self.is_log.get(col, False) else ""
                info[col]['details'] = (f"Applied {log_part}"
                                      f"differencing (order {info[col]['differencing_order']}).")
            elif self.method == 'boxcox':
                info[col]['details'] = (f"Applied Box-Cox transformation with lambda = {info[col]['boxcox_lambda']:.4f}.")
            elif self.method == 'log':
                skip_msg = " (skipped as data was in log form)" if self.is_log.get(col, False) else ""
                info[col]['details'] = f"Applied log transformation{skip_msg}."
            elif self.method == 'hp':
                info[col]['details'] = f"Applied Hodrick-Prescott filter with lambda = {self.lamb}."
            elif self.method == 'inverse':
                info[col]['details'] = f"Applied inverse transformation for {self.method}."
        
        return info