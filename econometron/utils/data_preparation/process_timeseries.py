import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.interpolate import interp1d
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress KPSS warnings

class TimeSeriesProcessor:
    """
    A class to process time series data: handle missing values, check stationarity,
    apply transformations, scale data, detect optimal lags, and visualize results.
    
    Attributes:
        data: pd.DataFrame, processed time series data
        stationary: dict, stationarity status for each column
        results: dict, detailed results including ADF/KPSS tests and transformations
    """
    
    def __init__(self, data, date_column=None, columns=None, max_diff=2, 
                 significance_level=0.05, plot=True, scale_type=None, max_lags=10,
                 stationarization_type=None, preferred_freq=None):
        """
        Initialize the TimeSeriesProcessor.

        Parameters:
        - data: DataFrame or Series with time series data
        - date_column: str, name of the date column (if DataFrame)
        - columns: list, columns to analyze (if None, all numeric columns are used)
        - max_diff: int, maximum differencing order (regular + seasonal)
        - significance_level: float, p-value threshold for stationarity tests
        - plot: bool, whether to plot series, ACF, and PACF
        - scale_type: str, 'standard' or 'minmax' for feature scaling (None for no scaling)
        - max_lags: int, maximum lags for ACF/PACF analysis
        - stationarization_type: str, preferred transformation ('log', 'diff', 'seasonal_diff', 
          'detrend', 'boxcox', 'log+diff')
        - preferred_freq: str, user-specified frequency ('M', 'Q', 'A', 'D') if index has none
        """
        self._validate_data(data)
        self.data = self._prepare_data(data, date_column, preferred_freq)
        self.columns = columns if columns else self.data.select_dtypes(include=[np.number]).columns
        self.max_diff = max_diff
        self.significance_level = significance_level
        self.plot = plot
        self.scale_type = scale_type
        self.max_lags = max_lags
        self.stationarization_type = stationarization_type
        self.stationary = {col: False for col in self.columns}
        self.results = {col: {'original': None, 'adf_results': {}, 'kpss_results': {}, 
                             'transformations': [], 'optimal_lag': None} 
                        for col in self.columns}
        self._process_columns()

    def _validate_data(self, data):
        """Check if all columns have the same length."""
        if isinstance(data, pd.DataFrame):
            lengths = [len(data[col]) for col in data.columns]
            if len(set(lengths)) > 1:
                raise ValueError("All columns must have the same length.")
        elif isinstance(data, pd.Series):
            pass
        else:
            raise ValueError("Input must be a pandas DataFrame or Series.")

    def _prepare_data(self, data, date_column, preferred_freq):
        """Prepare the input data by setting the index and inferring/setting frequency."""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if date_column:
            data = data.set_index(date_column)
            if not pd.api.types.is_datetime64_any_dtype(data.index):
                data.index = pd.to_datetime(data.index)
        
        # Infer or set frequency
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq:
            data.index.freq = inferred_freq
            print(f"Inferred frequency: {inferred_freq}")
        elif preferred_freq:
            try:
                data.index = pd.date_range(start=data.index[0], periods=len(data), freq=preferred_freq)
                data.index.freq = preferred_freq
                print(f"Set user-specified frequency: {preferred_freq}")
            except:
                raise ValueError(f"Invalid preferred frequency: {preferred_freq}. Use 'M', 'Q', 'A', or 'D'.")
        else:
            print("No frequency inferred and no preferred frequency provided. Setting to monthly as fallback.")
            data.index = pd.date_range(start=data.index[0], periods=len(data), freq='ME')
            data.index.freq = 'ME'
        return data

    def _impute_missing(self, series):
        """Handle missing values using linear interpolation and extrapolation."""
        if series.isna().any():
            print(f"Found {series.isna().sum()} missing values")
            series = series.interpolate(method='linear', limit_direction='both')
            if series.isna().any():
                non_na_idx = series.dropna().index
                if len(non_na_idx) > 1:
                    f = interp1d(non_na_idx.map(lambda x: x.timestamp()), series.dropna(),
                                 fill_value='extrapolate')
                    series = pd.Series(f(series.index.map(lambda x: x.timestamp())),
                                      index=series.index)
        return series

    def _adf_test(self, series, title=""):
        """Perform ADF test and return results."""
        result = adfuller(series.dropna(), autolag='AIC')
        return {'p_value': result[1], 'statistic': result[0], 'critical_values': result[4]}

    def _kpss_test(self, series, title=""):
        """Perform KPSS test and return results."""
        result = kpss(series.dropna(), regression='c', nlags='auto')
        return {'p_value': result[1], 'statistic': result[0], 'critical_values': result[3]}

    def _is_stationary(self, adf_result, kpss_result):
        """Determine stationarity based on ADF and KPSS tests."""
        return adf_result['p_value'] < self.significance_level and kpss_result['p_value'] > self.significance_level

    def _detrend(self, series):
        """Detrend series using linear or quadratic fit."""
        x = np.arange(len(series))
        coef = np.polyfit(x, series, 1)
        trend = np.polyval(coef, x)
        detrended = series - trend
        if not self._is_stationary(self._adf_test(detrended), self._kpss_test(detrended)):
            coef = np.polyfit(x, series, 2)
            trend = np.polyval(coef, x)
            detrended = series - trend
        return detrended, 'detrend'

    def _log_transform(self, series):
        """Apply log transformation (handle non-positive values)."""
        if (series <= 0).any():
            print("Non-positive values detected. Shifting data for log transform.")
            shift = abs(series.min()) + 1
            series = series + shift
        else:
            shift = 0
        return np.log(series), f'log (+{shift} if shifted)'

    def _seasonal_diff(self, series, period):
        """Apply seasonal differencing."""
        return series.diff(period).dropna(), f'seasonal_diff (period={period})'

    def _difference(self, series, order=1):
        """Apply regular differencing."""
        return series.diff(order).dropna(), f'diff (order={order})'

    def _boxcox_transform(self, series):
        """Apply Box-Cox transformation."""
        if (series <= 0).any():
            print("Non-positive values detected. Shifting data for Box-Cox.")
            shift = abs(series.min()) + 1
            series = series + shift
        else:
            shift = 0
        transformed, lam = boxcox(series)
        return pd.Series(transformed, index=series.index), f'boxcox (lambda={lam:.2f}, +{shift} if shifted)'

    def _log_diff(self, series):
        """Apply log transformation followed by differencing."""
        log_series, log_label = self._log_transform(series)
        diff_series, diff_label = self._difference(log_series)
        return diff_series, f'{log_label}+{diff_label}'

    def _scale_series(self, series):
        """Apply feature scaling (standard or minmax)."""
        if self.scale_type == 'standard':
            scaler = StandardScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), 'standard_scale'
        elif self.scale_type == 'minmax':
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=series.index), 'minmax_scale'
        return series, None

    def _find_optimal_lag(self, series):
        """Determine optimal lag using PACF."""
        pacf_vals, confint = pacf(series.dropna(), nlags=self.max_lags, alpha=0.05)
        for lag in range(1, len(pacf_vals)):
            if abs(pacf_vals[lag]) < confint[lag, 1] - pacf_vals[lag]:
                return lag
        return self.max_lags

    def _process_column(self, col):
        """Process a single column with transformations and stationarity checks."""
        series = self.data[col].copy()
        series = self._impute_missing(series)
        self.results[col]['original'] = series
        transformations = []

        # Plot original series
        if self.plot:
            plt.figure(figsize=(10, 4))
            plt.plot(series, label='Original')
            plt.title(f'Original Series - {col}')
            plt.legend()
            plt.show()

        # Check stationarity of original series
        adf_result = self._adf_test(series, "Original")
        kpss_result = self._kpss_test(series, "Original")
        self.results[col]['adf_results'][0] = adf_result
        self.results[col]['kpss_results'][0] = kpss_result

        if self._is_stationary(adf_result, kpss_result):
            print(f"{col} is stationary (ADF p-value: {adf_result['p_value']:.4f}, "
                  f"KPSS p-value: {kpss_result['p_value']:.4f})")
            self.stationary[col] = True
            transformed_series = series
        else:
            print(f"{col} is not stationary (ADF p-value: {adf_result['p_value']:.4f}, "
                  f"KPSS p-value: {kpss_result['p_value']:.4f})")
            
            # Define transformation options
            transformation_map = {
                'detrend': self._detrend,
                'log': self._log_transform,
                'log+diff': self._log_diff,
                'boxcox': self._boxcox_transform,
                'diff': self._difference
            }
            if self.data.index.inferred_freq:
                period = {'ME': 12, 'Q': 4, 'A': 1, 'D': 7}.get(self.data.index.inferred_freq, 12)
                transformation_map['seasonal_diff'] = lambda s: self._seasonal_diff(s, period)

            # Apply user-specified transformation if provided
            if self.stationarization_type and self.stationarization_type in transformation_map:
                print(f"Applying user-specified transformation: {self.stationarization_type}")
                transform = transformation_map[self.stationarization_type]
                transformed_series, transform_label = transform(series)
                transformations.append(transform_label)
                adf_result = self._adf_test(transformed_series, self.stationarization_type)
                kpss_result = self._kpss_test(transformed_series, self.stationarization_type)
                self.results[col]['adf_results'][self.stationarization_type] = adf_result
                self.results[col]['kpss_results'][self.stationarization_type] = kpss_result

                if self._is_stationary(adf_result, kpss_result):
                    print(f"{col} becomes stationary after {transform_label} "
                          f"(ADF p-value: {adf_result['p_value']:.4f}, "
                          f"KPSS p-value: {kpss_result['p_value']:.4f})")
                    self.stationary[col] = True
                else:
                    print(f"{col} is not stationary after {transform_label} "
                          f"(ADF p-value: {adf_result['p_value']:.4f}, "
                          f"KPSS p-value: {kpss_result['p_value']:.4f})")
                    # Try additional differencing if needed
                    for diff_order in range(2, self.max_diff + 1):
                        transformed_series, transform_label = self._difference(transformed_series, diff_order)
                        transformations.append(transform_label)
                        adf_result = self._adf_test(transformed_series, f'diff_order_{diff_order}')
                        kpss_result = self._kpss_test(transformed_series, f'diff_order_{diff_order}')
                        self.results[col]['adf_results'][diff_order] = adf_result
                        self.results[col]['kpss_results'][diff_order] = kpss_result
                        if self._is_stationary(adf_result, kpss_result):
                            print(f"{col} becomes stationary after additional differencing {diff_order} "
                                  f"(ADF p-value: {adf_result['p_value']:.4f}, "
                                  f"KPSS p-value: {kpss_result['p_value']:.4f})")
                            self.stationary[col] = True
                            break
            else:
                # Try transformations in order of complexity
                transformations_to_try = [
                    ('detrend', self._detrend),
                    ('log', self._log_transform),
                    ('log+diff', self._log_diff),
                    ('boxcox', self._boxcox_transform),
                    ('diff', self._difference)
                ]
                if self.data.index.inferred_freq:
                    period = {'M': 12, 'Q': 4, 'A': 1, 'D': 7}.get(self.data.index.inferred_freq, 12)
                    transformations_to_try.insert(2, ('seasonal_diff', lambda s: self._seasonal_diff(s, period)))

                for name, transform in transformations_to_try:
                    transformed_series, transform_label = transform(series)
                    transformations.append(transform_label)
                    adf_result = self._adf_test(transformed_series, name)
                    kpss_result = self._kpss_test(transformed_series, name)
                    self.results[col]['adf_results'][name] = adf_result
                    self.results[col]['kpss_results'][name] = kpss_result

                    if self._is_stationary(adf_result, kpss_result):
                        print(f"{col} becomes stationary after {transform_label} "
                              f"(ADF p-value: {adf_result['p_value']:.4f}, "
                              f"KPSS p-value: {kpss_result['p_value']:.4f})")
                        self.stationary[col] = True
                        break
                else:
                    for diff_order in range(2, self.max_diff + 1):
                        transformed_series, transform_label = self._difference(transformed_series, diff_order)
                        transformations.append(transform_label)
                        adf_result = self._adf_test(transformed_series, f'diff_order_{diff_order}')
                        kpss_result = self._kpss_test(transformed_series, f'diff_order_{diff_order}')
                        self.results[col]['adf_results'][diff_order] = adf_result
                        self.results[col]['kpss_results'][diff_order] = kpss_result
                        if self._is_stationary(adf_result, kpss_result):
                            print(f"{col} becomes stationary after additional differencing {diff_order} "
                                  f"(ADF p-value: {adf_result['p_value']:.4f}, "
                                  f"KPSS p-value: {kpss_result['p_value']:.4f})")
                            self.stationary[col] = True
                            break
                    else:
                        print(f"{col} is not stationary after maximum transformations.")

        # Store transformed series
        self.results[col]['differenced'] = transformed_series
        self.results[col]['transformations'] = transformations

        # Apply scaling if specified
        if self.scale_type:
            transformed_series, scale_label = self._scale_series(transformed_series)
            if scale_label:
                transformations.append(scale_label)
                self.results[col]['differenced'] = transformed_series

        # Determine optimal lag
        self.results[col]['optimal_lag'] = self._find_optimal_lag(transformed_series)

        # Plot transformed series and ACF/PACF
        if self.plot and self.stationary[col]:
            plt.figure(figsize=(12, 8))
            plt.subplot(311)
            plt.plot(transformed_series, label='Transformed')
            plt.title(f'Transformed Series - {col} ({", ".join(transformations)})')
            plt.legend()

            plt.subplot(312)
            plot_acf(transformed_series, lags=self.max_lags, ax=plt.gca())
            plt.title(f'ACF - {col}')

            plt.subplot(313)
            plot_pacf(transformed_series, lags=self.max_lags, ax=plt.gca())
            plt.title(f'PACF - {col} (Optimal lag: {self.results[col]["optimal_lag"]})')

            plt.tight_layout()
            plt.show()

    def _process_columns(self):
        """Process all specified columns."""
        for col in self.columns:
            print(f"\nProcessing column: {col}")
            self._process_column(col)

    def get_results(self):
        """Return the processing results."""
        return self.results

    def get_stationary_data(self):
        """Return a DataFrame with stationary (transformed) series."""
        return pd.DataFrame({col: self.results[col]['differenced'] for col in self.columns})