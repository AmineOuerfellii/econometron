import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesProcessor:

    def __init__(self, data, columns=None, method='diff', demean=True, analysis=True, plot=False, lamb=1600, log_data=True):
        """
        Initialize TimeSeriesProcessor.
        
        Parameters:
        - data: pandas DataFrame or numpy array
        - columns: list of column names to process (if None, all numeric columns are processed)
        - method: transformation method ('diff', 'log_diff', 'detrend', 'seasonal', 'boxcox', 'hodrick_prescott')
        - demean: bool, whether to demean the data before processing
        - analysis: bool, whether to perform stationarity analysis
        - plot: bool, whether to generate ACF/PACF plots
        - lamb: lambda parameter for Hodrick-Prescott filter
        - log_data: bool, whether to apply log transformation before HP filter (only for hodrick_prescott method)
        """
        # Store original data before any modifications
        self.original_data = pd.DataFrame(data, columns=columns) if isinstance(data, np.ndarray) else data.copy()
        
        # Working copy
        self.data = self.original_data.copy()
        
        if columns is None:
            self.columns = [col for col in self.data.columns if pd.api.types.is_numeric_dtype(self.data[col])]
        else:
            self.columns = columns
            
        self.method = method
        self.analysis = analysis
        self.plot = plot
        self.lamb = lamb if lamb is not None else 1600
        self.log_data = log_data
        self.demean = demean
        
        # Storage for results and parameters
        self.stationary_info = {col: {} for col in self.columns}
        self.transformed_data = None
        self.transform_params = {col: {} for col in self.columns}
        self.means = {}  # Store means for demeaning
        
        # Validation
        self._validate_data()
        
        # Apply demeaning if requested (but be more careful)
        if self.demean:
            self._apply_demeaning()
        
        # Perform analysis and transformation
        if self.analysis:
            self._analyze_stationarity()

    def _validate_data(self):
        """Validate input data with more stringent checks."""
        if not all(pd.api.types.is_numeric_dtype(self.data[col]) for col in self.columns):
            raise ValueError("Not all specified columns are numeric")
        
        for col in self.columns:
            clean_data = self.data[col].dropna()
            if len(clean_data) < 20:  # Increased minimum requirement
                raise ValueError(f"Column {col} has insufficient data points (need at least 20, got {len(clean_data)})")
            if clean_data.nunique() <= 2:  # More strict uniqueness requirement
                raise ValueError(f"Column {col} has insufficient unique values (need more than 2, got {clean_data.nunique()})")
            
            # Check for obvious non-stationarity patterns
            if len(clean_data) >= 10:
                # Check for monotonic trends
                diff_sign_changes = np.sum(np.diff(np.sign(np.diff(clean_data.values))) != 0)
                if diff_sign_changes < len(clean_data) * 0.1:  # Less than 10% sign changes
                    print(f"Warning: Column {col} may have strong trend (few direction changes)")
                
                # Check for variance explosion
                if clean_data.std() > 3 * clean_data.abs().median():
                    print(f"Warning: Column {col} may have high variance instability")

    def _apply_demeaning(self):
        """Apply demeaning and store means for reversal."""
        for col in self.columns:
            self.means[col] = self.data[col].mean()
            self.data[col] = self.data[col] - self.means[col]
            
            # Check if demeaning made data constant
            if self.data[col].dropna().nunique() <= 1:
                raise ValueError(f"Column {col} became constant after demeaning")

    def _check_stationarity(self, series, col_name, update_info=True, significance_level=0.05):
        """Perform comprehensive stationarity tests."""
        series_clean = series.dropna()
        
        # Basic data validation
        if len(series_clean) < 20:  # Increased minimum requirement
            if update_info:
                self.stationary_info[col_name].update({
                    'is_stationary': False,
                    'p_value': np.nan,
                    'adf_statistic': np.nan,
                    'error': f'Insufficient data (n={len(series_clean)}, need at least 20)'
                })
            return False
            
        if series_clean.nunique() <= 2:  # More strict uniqueness check
            if update_info:
                self.stationary_info[col_name].update({
                    'is_stationary': False,
                    'p_value': np.nan,
                    'adf_statistic': np.nan,
                    'error': f'Insufficient variation (only {series_clean.nunique()} unique values)'
                })
            return False
        
        # Check for obvious trends using simple linear regression
        if len(series_clean) >= 10:
            x = np.arange(len(series_clean))
            try:
                slope, intercept = np.polyfit(x, series_clean.values, 1)
                # Normalize slope by data range to get relative trend
                data_range = series_clean.max() - series_clean.min()
                if data_range > 0:
                    relative_slope = abs(slope * len(series_clean)) / data_range
                    # If trend is more than 10% of data range, likely non-stationary
                    if relative_slope > 0.1:
                        if update_info:
                            self.stationary_info[col_name].update({
                                'is_stationary': False,
                                'p_value': 1.0,  # Fail the test
                                'adf_statistic': np.nan,
                                'error': f'Strong linear trend detected (relative slope: {relative_slope:.3f})'
                            })
                        return False
            except:
                pass  # Continue with ADF test if trend detection fails
        
        # Check for variance changes (heteroscedasticity)
        if len(series_clean) >= 20:
            try:
                mid_point = len(series_clean) // 2
                first_half_var = series_clean.iloc[:mid_point].var()
                second_half_var = series_clean.iloc[mid_point:].var()
                
                # If variance ratio is too extreme, likely non-stationary
                if first_half_var > 0 and second_half_var > 0:
                    var_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
                    if var_ratio > 10:  # Variance changes by factor of 10+
                        if update_info:
                            self.stationary_info[col_name].update({
                                'is_stationary': False,
                                'p_value': 1.0,
                                'adf_statistic': np.nan,
                                'error': f'High variance instability (ratio: {var_ratio:.2f})'
                            })
                        return False
            except:
                pass
        
        # Enhanced ADF test with multiple specifications
        try:
            # Test with different regression types
            test_results = []
            
            for regression in ['c', 'ct', 'ctt']:  # constant, constant+trend, constant+trend+trend^2
                try:
                    max_lags = min(12, len(series_clean) // 5)  # More conservative lag selection
                    if max_lags < 1:
                        max_lags = 1
                        
                    result = adfuller(
                        series_clean, 
                        maxlag=max_lags,
                        regression=regression,
                        autolag='AIC'
                    )
                    test_results.append({
                        'regression': regression,
                        'adf_stat': result[0],
                        'p_value': result[1],
                        'critical_values': result[4]
                    })
                except:
                    continue
            
            if not test_results:
                if update_info:
                    self.stationary_info[col_name].update({
                        'is_stationary': False,
                        'p_value': np.nan,
                        'adf_statistic': np.nan,
                        'error': 'All ADF test specifications failed'
                    })
                return False
            
            # Use the most conservative result (highest p-value)
            best_result = max(test_results, key=lambda x: x['p_value'] if not np.isnan(x['p_value']) else 1.0)
            
            # Additional check: ADF statistic should be significantly negative
            # compared to critical values
            adf_stat = best_result['adf_stat']
            critical_1pct = best_result['critical_values'].get('1%', -3.43)
            critical_5pct = best_result['critical_values'].get('5%', -2.86)
            
            # Require ADF statistic to be more negative than 5% critical value
            # and p-value to be below significance level
            p_value = best_result['p_value']
            is_stationary = (p_value < significance_level and 
                           adf_stat < critical_5pct and
                           not np.isnan(p_value))
            
            if update_info:
                self.stationary_info[col_name].update({
                    'is_stationary': is_stationary,
                    'p_value': p_value,
                    'adf_statistic': adf_stat,
                    'critical_5pct': critical_5pct,
                    'regression_type': best_result['regression'],
                    'n_observations': len(series_clean)
                })
                
            return is_stationary
            
        except Exception as e:
            if update_info:
                self.stationary_info[col_name].update({
                    'is_stationary': False,
                    'p_value': np.nan,
                    'adf_statistic': np.nan,
                    'error': f'ADF test failed: {str(e)}',
                    'n_observations': len(series_clean)
                })
            return False

    def _apply_differencing(self, series, col_name, max_diff=3):
        """Apply differencing transformation with proper tracking."""
        current_series = series.copy()
        diff_order = 0
        original_values = []  # Store first values for reconstruction
        
        # Check if already stationary
        if self._check_stationarity(current_series, col_name, update_info=False):
            self.transform_params[col_name] = {
                'method': 'diff',
                'order': 0,
                'first_values': []
            }
            return current_series
        
        # Apply differencing
        while diff_order < max_diff:
            # Store the first non-NaN value before differencing
            first_val = current_series.dropna().iloc[0] if len(current_series.dropna()) > 0 else 0
            original_values.append(first_val)
            
            # Apply differencing
            current_series = current_series.diff().dropna()
            diff_order += 1
            
            # Check data validity after differencing
            if len(current_series.dropna()) < 20:  # Increased requirement
                self.stationary_info[col_name]['error'] = f'Insufficient data after {diff_order} differences'
                break
                
            if current_series.dropna().nunique() <= 1:
                self.stationary_info[col_name]['error'] = f'Series became constant after {diff_order} differences'
                break
                
            # Check stationarity
            if self._check_stationarity(current_series, col_name, update_info=False):
                break
        
        # Store transformation parameters
        self.transform_params[col_name] = {
            'method': 'diff',
            'order': diff_order,
            'first_values': original_values
        }
        
        return current_series

    def _apply_log_differencing(self, series, col_name, max_diff=3):
        """Apply log + differencing transformation."""
        # Handle non-positive values
        min_val = series.min()
        shift_val = 0
        if min_val <= 0:
            shift_val = abs(min_val) + 1e-6
            
        current_series = np.log(series + shift_val)
        
        # Check for invalid values
        if current_series.isnull().any() or np.isinf(current_series).any():
            self.stationary_info[col_name]['error'] = 'Log transformation produced invalid values'
            return series
        
        # Apply differencing on log-transformed data
        diff_order = 0
        original_values = []
        
        # Check if log-transformed data is already stationary
        if self._check_stationarity(current_series, col_name, update_info=False):
            self.transform_params[col_name] = {
                'method': 'log_diff',
                'order': 0,
                'first_values': [],
                'shift_val': shift_val
            }
            return current_series
        
        # Apply differencing
        while diff_order < max_diff:
            first_val = current_series.dropna().iloc[0] if len(current_series.dropna()) > 0 else 0
            original_values.append(first_val)
            
            current_series = current_series.diff().dropna()
            diff_order += 1
            
            if len(current_series.dropna()) < 20:
                self.stationary_info[col_name]['error'] = f'Insufficient data after {diff_order} log-differences'
                break
                
            if current_series.dropna().nunique() <= 1:
                self.stationary_info[col_name]['error'] = f'Series became constant after {diff_order} log-differences'
                break
                
            if self._check_stationarity(current_series, col_name, update_info=False):
                break
        
        self.transform_params[col_name] = {
            'method': 'log_diff',
            'order': diff_order,
            'first_values': original_values,
            'shift_val': shift_val
        }
        
        return current_series

    def _apply_detrending(self, series, col_name, window=12):
        """Apply detrending transformation."""
        try:
            trend = series.rolling(window=window, center=True).mean()
            detrended = series - trend
            
            self.transform_params[col_name] = {
                'method': 'detrend',
                'trend': trend,
                'window': window
            }
            
            return detrended.dropna()
            
        except Exception as e:
            self.stationary_info[col_name]['error'] = f'Detrending failed: {str(e)}'
            return series

    def _apply_seasonal_decomposition(self, series, col_name, period=12):
        """Apply seasonal decomposition."""
        try:
            if len(series.dropna()) < 2 * period:
                self.stationary_info[col_name]['error'] = f'Insufficient data for seasonal decomposition (need at least {2*period} points)'
                return series
                
            decomposition = seasonal_decompose(
                series.dropna(), 
                period=period, 
                model='additive', 
                extrapolate_trend='freq'
            )
            
            self.transform_params[col_name] = {
                'method': 'seasonal',
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'period': period
            }
            
            return decomposition.resid.dropna()
            
        except Exception as e:
            self.stationary_info[col_name]['error'] = f'Seasonal decomposition failed: {str(e)}'
            return series

    def _apply_boxcox(self, series, col_name):
        """Apply Box-Cox transformation."""
        try:
            # Handle non-positive values
            min_val = series.min()
            shift_val = 0
            if min_val <= 0:
                shift_val = abs(min_val) + 1e-6
                
            transformed, lmbda = boxcox(series + shift_val)
            
            self.transform_params[col_name] = {
                'method': 'boxcox',
                'lambda': lmbda,
                'shift_val': shift_val
            }
            
            return pd.Series(transformed, index=series.index)
            
        except Exception as e:
            self.stationary_info[col_name]['error'] = f'Box-Cox transformation failed: {str(e)}'
            return series

    def _apply_hodrick_prescott(self, series, col_name):
        """Apply Hodrick-Prescott filter with optional log transformation."""
        try:
            if self.log_data:
                # Handle non-positive values for log transformation
                min_val = series.min()
                shift_val = 0
                if min_val <= 0:
                    shift_val = abs(min_val) + 1e-6
                    
                log_series = np.log(series + shift_val)
                
                # Check for invalid values after log transformation
                if log_series.isnull().any() or np.isinf(log_series).any():
                    self.stationary_info[col_name]['error'] = 'Log transformation produced invalid values'
                    return series
                
                cycle, trend = hpfilter(log_series.dropna(), lamb=self.lamb)
                
                self.transform_params[col_name] = {
                    'method': 'hodrick_prescott',
                    'trend': trend,
                    'shift_val': shift_val,
                    'lambda': self.lamb,
                    'log_applied': True
                }
            else:
                # Apply HP filter directly without log transformation
                cycle, trend = hpfilter(series.dropna(), lamb=self.lamb)
                
                self.transform_params[col_name] = {
                    'method': 'hodrick_prescott',
                    'trend': trend,
                    'shift_val': 0,
                    'lambda': self.lamb,
                    'log_applied': False
                }
            
            return cycle
            
        except Exception as e:
            self.stationary_info[col_name]['error'] = f'Hodrick-Prescott filter failed: {str(e)}'
            return series

    def _analyze_stationarity(self):
        """Analyze stationarity and apply transformations."""
        self.transformed_data = pd.DataFrame(index=self.data.index)
        
        for col in self.columns:
            print(f"Processing column: {col}")
            
            # Initialize column info
            self.stationary_info[col] = {'transformation_applied': self.method}
            
            # Check original stationarity
            original_stationary = self._check_stationarity(self.data[col], col, update_info=False)
            
            if original_stationary:
                print(f"  {col} is already stationary")
                self.transformed_data[col] = self.data[col]
                self.transform_params[col] = {'method': 'none', 'order': 0}
                self.stationary_info[col]['is_stationary'] = True
                continue
            
            # Apply transformation based on method
            if self.method == 'diff':
                transformed_series = self._apply_differencing(self.data[col], col)
            elif self.method == 'log_diff':
                transformed_series = self._apply_log_differencing(self.data[col], col)
            elif self.method == 'detrend':
                transformed_series = self._apply_detrending(self.data[col], col)
            elif self.method == 'seasonal':
                transformed_series = self._apply_seasonal_decomposition(self.data[col], col)
            elif self.method == 'boxcox':
                transformed_series = self._apply_boxcox(self.data[col], col)
            elif self.method == 'hodrick_prescott':
                transformed_series = self._apply_hodrick_prescott(self.data[col], col)
            else:
                raise ValueError(f"Unknown transformation method: {self.method}")
            
            # Store transformed data
            self.transformed_data[col] = transformed_series
            
            # Final stationarity check
            if 'error' not in self.stationary_info[col]:
                final_stationary = self._check_stationarity(transformed_series, col, update_info=True)
                print(f"  {col} final stationarity: {final_stationary}")
            else:
                print(f"  {col} transformation failed: {self.stationary_info[col]['error']}")
            
            # Plot if requested
            if self.plot and 'error' not in self.stationary_info[col]:
                self._plot_acf_pacf(transformed_series, col)

    def _plot_acf_pacf(self, series, col_name):
        """Plot ACF and PACF for a given series."""
        series_clean = series.dropna()
        if len(series_clean) < 10:
            return
        
        nlags = min(20, len(series_clean) // 4)
        if nlags < 1:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        try:
            # ACF plot
            acf_vals = acf(series_clean, nlags=nlags, fft=True)
            ax1.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
            ax1.set_title(f'ACF - {col_name}')
            ax1.set_xlabel('Lags')
            ax1.set_ylabel('ACF')
            
            # PACF plot
            pacf_vals = pacf(series_clean, nlags=nlags)
            ax2.stem(range(len(pacf_vals)), pacf_vals, basefmt=" ")
            ax2.set_title(f'PACF - {col_name}')
            ax2.set_xlabel('Lags')
            ax2.set_ylabel('PACF')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed for {col_name}: {str(e)}")

    def get_stationary_info(self):
        """Return stationarity analysis results."""
        return self.stationary_info

    def get_transformed_data(self):
        """Return transformed data."""
        return self.transformed_data.dropna()

    def get_original_data(self):
        """Return original data before any transformations."""
        return self.original_data

    def untransform(self, data=None, columns=None):
        """
        Inverse transform the data back to original scale.
        
        Parameters:
        - data: DataFrame to untransform (if None, uses transformed_data)
        - columns: specific columns to untransform (if None, untransform all)
        """
        if data is None:
            if self.transformed_data is None:
                raise ValueError("No transformed data available")
            data = self.transformed_data.copy()
        else:
            data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data.copy()

        if columns is None:
            columns = self.columns
        else:
            columns = [col for col in columns if col in self.columns]

        result = data.copy()
        
        for col in columns:
            if col not in self.transform_params:
                continue
                
            params = self.transform_params[col]
            method = params.get('method')
            
            print(f"Untransforming {col} using method: {method}")
            
            try:
                if method == 'none':
                    # No transformation was applied
                    pass
                    
                elif method == 'diff':
                    result[col] = self._inverse_differencing(
                        data[col], 
                        params.get('order', 0),
                        params.get('first_values', []),
                        col
                    )
                    
                elif method == 'log_diff':
                    # First inverse differencing
                    log_series = self._inverse_differencing(
                        data[col],
                        params.get('order', 0),
                        params.get('first_values', []),
                        col
                    )
                    # Then inverse log
                    shift_val = params.get('shift_val', 0)
                    result[col] = np.exp(log_series) - shift_val
                    
                elif method == 'detrend':
                    trend = params.get('trend')
                    if trend is not None:
                        # Align indices
                        common_idx = data[col].index.intersection(trend.index)
                        result.loc[common_idx, col] = data.loc[common_idx, col] + trend.loc[common_idx]
                        
                elif method == 'seasonal':
                    trend = params.get('trend')
                    seasonal = params.get('seasonal')
                    if trend is not None and seasonal is not None:
                        common_idx = data[col].index.intersection(trend.index).intersection(seasonal.index)
                        result.loc[common_idx, col] = (data.loc[common_idx, col] + 
                                                     trend.loc[common_idx] + 
                                                     seasonal.loc[common_idx])
                        
                elif method == 'boxcox':
                    lmbda = params.get('lambda')
                    shift_val = params.get('shift_val', 0)
                    if lmbda is not None:
                        result[col] = self._inverse_boxcox(data[col], lmbda) - shift_val
                        
                elif method == 'hodrick_prescott':
                    trend = params.get('trend')
                    shift_val = params.get('shift_val', 0)
                    log_applied = params.get('log_applied', False)
                    
                    if trend is not None:
                        common_idx = data[col].index.intersection(trend.index)
                        
                        if log_applied:
                            # Add back trend in log space, then exp, then remove shift
                            log_reconstructed = data.loc[common_idx, col] + trend.loc[common_idx]
                            result.loc[common_idx, col] = np.exp(log_reconstructed) - shift_val
                        else:
                            # Add back trend directly (no log transformation was applied)
                            result.loc[common_idx, col] = data.loc[common_idx, col] + trend.loc[common_idx]
                
                # Add back the mean if demeaning was applied
                if self.demean and col in self.means:
                    result[col] = result[col] + self.means[col]
                    
            except Exception as e:
                print(f"Failed to untransform {col}: {str(e)}")
                result[col] = data[col]  # Keep transformed data if untransform fails

        return result

    def _inverse_boxcox(self, series, lmbda):
        """Inverse Box-Cox transformation."""
        if lmbda == 0:
            return np.exp(series)
        else:
            return np.sign(series * lmbda + 1) * np.abs(series * lmbda + 1) ** (1 / lmbda)

    def _inverse_differencing(self, diff_series, order, first_values, col_name):
        """Properly inverse differencing transformation."""
        if order == 0:
            return diff_series
            
        result = diff_series.copy()
        
        # Get the original series to use as reference
        original_series = self.data[col_name] if self.demean else self.original_data[col_name]
        
        # Apply inverse differencing for each order
        for i in range(order):
            if i < len(first_values):
                # Use stored first value
                first_val = first_values[i]
            else:
                # Fallback to original data
                first_val = original_series.iloc[i] if len(original_series) > i else 0
            
            # Reconstruct the series
            result = result.fillna(0).cumsum() + first_val
            
        return result

    def summary(self):
        """Print a summary of the transformation results."""
        print("="*60)
        print("TIME SERIES PROCESSOR SUMMARY")
        print("="*60)
        print(f"Method: {self.method}")
        print(f"Demean: {self.demean}")
        print(f"Columns processed: {len(self.columns)}")
        print()
        
        for col in self.columns:
            info = self.stationary_info.get(col, {})
            params = self.transform_params.get(col, {})
            
            print(f"Column: {col}")
            print(f"  Transformation: {params.get('method', 'none')}")
            
            if params.get('method') == 'diff':
                print(f"  Differencing order: {params.get('order', 0)}")
            elif params.get('method') == 'log_diff':
                print(f"  Log-differencing order: {params.get('order', 0)}")
                print(f"  Shift value: {params.get('shift_val', 0):.6f}")
            elif params.get('method') == 'boxcox':
                print(f"  Lambda: {params.get('lambda', 'N/A')}")
            elif params.get('method') == 'hodrick_prescott':
                print(f"  Lambda: {params.get('lambda', 'N/A')}")
                print(f"  Log applied: {params.get('log_applied', 'N/A')}")
                if params.get('log_applied'):
                    print(f"  Shift value: {params.get('shift_val', 0):.6f}")
            
            if 'error' in info:
                print(f"  Status: ERROR - {info['error']}")
            else:
                print(f"  Stationary: {info.get('is_stationary', 'Unknown')}")
                print(f"  ADF p-value: {info.get('p_value', 'N/A'):.6f}" if info.get('p_value') is not None else "  ADF p-value: N/A")
                print(f"  Observations: {info.get('n_observations', 'N/A')}")
            print()