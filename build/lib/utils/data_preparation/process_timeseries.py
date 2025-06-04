import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
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