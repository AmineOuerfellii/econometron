import numpy as np
import pandas as pd
import logging
from typing import Union, List, Optional
import econometron.Models.VectorAutoReg.VARMA as VARMA
import econometron.utils.data_preparation.process_timeseries as TransformTS
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VARIMA(VARMA.VARMA_):
    """
    VARIMA model for multivariate time series forecasting with integrated differencing.

    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Input time series data.
    max_p : int, optional
        Maximum order of the autoregressive part (default: 5).
    max_d : int, optional
        Maximum order of differencing (default: 2).
    max_q : int, optional
        Maximum order of the moving average part (default: 5).
    columns : list, optional
        Columns to use from the DataFrame. If None, all numeric columns are selected.
    forecast_h : int, optional
        Forecast horizon (default: 6).
    plot : bool, optional
        Whether to plot the results (default: True).
    check_stationarity : bool, optional
        Whether to check for stationarity (default: True).
    bootstrap_n : int, optional
        Number of bootstrap samples for confidence intervals (default: 1000).
    criterion : str, optional
        Information criterion for model selection ('AIC' or 'BIC', default: 'AIC').
    ci_alpha : float, optional
        Significance level for confidence intervals (default: 0.05).
    Key : str, optional
        Key for identifying the model in results (default: None).
    Threshold : float, optional
        Threshold for determining significant parameters (default: 0.8).
    orth : bool, optional
        Whether to orthogonalize the residuals (default: False).
    log_data : bool, optional
        Whether to apply log transformation for non-positive data (default: True).
    max_diff : int, optional
        Maximum differencing order before switching to log-diff (default: 2).

    Attributes:
    -----------
    data : pd.DataFrame
        Transformed (differenced) time series data.
    original_data : pd.DataFrame
        Original input data before differencing.
    I : TransformTS
        Instance of TransformTS for data transformation and stationarity checks.
    diff_orders : dict
        Dictionary storing differencing orders for each column.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray], max_p: int = 5, max_d: int = 2, max_q: int = 5,
                 columns: Optional[List[str]] = None, forecast_h: int = 6, plot: bool = True,
                 check_stationarity: bool = True, bootstrap_n: int = 1000, criterion: str = 'AIC',
                 ci_alpha: float = 0.05, Key: Optional[str] = None, Threshold: float = 0.8, orth: bool = False,
                 log_data: bool = True, max_diff: int = 2):
        """
        Initialize the VARIMA model with differencing and transformation handling.
        """
        # Convert numpy array to DataFrame if necessary
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Initialize TransformTS with differencing method
        self.I = TransformTS(data=data, columns=columns, method='diff', analysis=check_stationarity,
                             plot=False, log_data=log_data, max_diff=max_diff)
        self.original_data = data[self.I.columns].copy()
        self.diff_orders = self.I.diff_order  # Store differencing orders from TransformTS
        
        # Get transformed (differenced) data
        self.data = self.I.get_transformed_data()
        
        # Validate differencing orders
        if max_d < max(self.diff_orders.values()):
            logger.warning(f"max_d ({max_d}) is less than the maximum differencing order "
                          f"({max(self.diff_orders.values())}) required for stationarity. Setting max_d to "
                          f"{max(self.diff_orders.values())}.")
            max_d = max(self.diff_orders.values())
        
        # Initialize parent VARMA class with transformed data
        super().__init__(data=self.data, max_p=max_p, max_q=max_q, columns=columns, forecast_h=forecast_h,
                         plot=plot, check_stationarity=check_stationarity, bootstrap_n=bootstrap_n,
                         criterion=criterion, ci_alpha=ci_alpha, Key=Key, Threshold=Threshold, orth=orth,
                         structural_id=False)  # structural_id set to False as VARIMA handles integration
        self.max_d = max_d

    def fit(self, p: Optional[int] = None, q: Optional[int] = None, output: bool = True) -> dict:
        """
        Fit the VARIMA model using the parent VARMA fit method and store results.

        Parameters:
        -----------
        p : int, optional
            Order of the autoregressive part.
        q : int, optional
            Order of the moving average part.
        output : bool, optional
            Whether to print results and diagnostics (default: True).

        Returns:
        --------
        dict
            Dictionary containing model fit results, including parameters, residuals, and diagnostics.
        """
        # Call parent VARMA fit method
        self.best_model = super().fit(p=p, q=q, output=output)
        
        # Ensure differencing orders are stored
        self.diff_orders = self.I.diff_order
        
        if output:
            logger.info("VARIMA model fitting completed.")
            print(f"VARIMA({self.best_p},{max(self.diff_orders.values())},{self.best_q})")
            self.trns_info()
        
        return self.best_model

    def trns_info(self) -> dict:
        """
        Retrieve transformation and stationarity information for each column.

        Returns:
        --------
        dict
            A dictionary containing transformation details, differencing order,
            and stationarity information for each column.
        """
        info = self.I.trns_info()
        print("\n=== Transformation Information ===")
        for col, details in info.items():
            print(f"\nColumn: {col}")
            for key, value in details.items():
                print(f"  {key}: {value}")
        return info

    def predict(self, n_periods: int = 6, plot: bool = True, tol: float = 1e-6) -> dict:
        """
        Generate forecasts and reverse differencing to return predictions in original scale.

        Parameters:
        -----------
        n_periods : int, optional
            Number of periods to forecast (default: 6).
        plot : bool, optional
            Whether to plot forecasts (default: True).
        tol : float, optional
            Tolerance for numerical stability (default: 1e-6).

        Returns:
        --------
        dict
            Dictionary containing point forecasts, lower and upper confidence intervals
            in the original scale.
        """
        if self.best_model is None:
            raise ValueError("No model fitted. Cannot generate forecasts.")

        # Generate forecasts in differenced scale using parent VARMA predict method
        forecasts = super().predict(n_periods=n_periods, plot=False, tol=tol)
        forecast_df = forecasts['point']
        ci_lower_df = forecasts['ci_lower']
        ci_upper_df = forecasts['ci_upper']
        
        # Reverse differencing to return to original scale
        K = len(self.columns)
        T = len(self.original_data)
        forecast_orig = np.zeros((n_periods, K))
        ci_lower_orig = np.zeros((n_periods, K))
        ci_upper_orig = np.zeros((n_periods, K))
        
        for i, col in enumerate(self.columns):
            d = self.diff_orders.get(col, 0)
            last_values = self.original_data[col].values[-d:] if d > 0 else []
            
            # Handle log transformation if applied
            is_log = self.I.is_log.get(col, False)
            if is_log:
                forecast_df[col] = np.exp(forecast_df[col])
                ci_lower_df[col] = np.exp(ci_lower_df[col])
                ci_upper_df[col] = np.exp(ci_upper_df[col])
            
            # Reverse differencing
            if d == 0:
                forecast_orig[:, i] = forecast_df[col].values
                ci_lower_orig[:, i] = ci_lower_df[col].values
                ci_upper_orig[:, i] = ci_upper_df[col].values
            else:
                # Initialize with last observed values
                cumsum = np.zeros(n_periods)
                for j in range(n_periods):
                    if j < d:
                        cumsum[j] = last_values[-d + j] if j < len(last_values) else 0
                    if j > 0:
                        cumsum[j] += cumsum[j - 1]
                    cumsum[j] += forecast_df[col].values[j]
                forecast_orig[:, i] = cumsum
                
                # Confidence intervals
                cumsum_lower = np.zeros(n_periods)
                cumsum_upper = np.zeros(n_periods)
                for j in range(n_periods):
                    if j < d:
                        cumsum_lower[j] = last_values[-d + j] if j < len(last_values) else 0
                        cumsum_upper[j] = last_values[-d + j] if j < len(last_values) else 0
                    if j > 0:
                        cumsum_lower[j] += cumsum_lower[j - 1]
                        cumsum_upper[j] += cumsum_upper[j - 1]
                    cumsum_lower[j] += ci_lower_df[col].values[j]
                    cumsum_upper[j] += ci_upper_df[col].values[j]
                ci_lower_orig[:, i] = cumsum_lower
                ci_upper_orig[:, i] = cumsum_upper
        
        # Create DataFrames for output
        forecast_dates = pd.date_range(
            start=self.original_data.index[-1] + pd.Timedelta(days=1),
            periods=n_periods,
            freq=self.original_data.index.inferred_freq or 'D'
        ) if isinstance(self.original_data.index, pd.DatetimeIndex) else range(T, T + n_periods)
        
        forecast_df_orig = pd.DataFrame(forecast_orig, index=forecast_dates, columns=self.columns)
        ci_lower_df_orig = pd.DataFrame(ci_lower_orig, index=forecast_dates, columns=self.columns)
        ci_upper_df_orig = pd.DataFrame(ci_upper_orig, index=forecast_dates, columns=self.columns)
        
        # Plotting
        if plot:
            n_vars = len(self.columns)
            n_cols = min(2, n_vars)
            n_rows = (n_vars + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
            axes = np.array(axes).flatten() if n_vars > 1 else [axes]
            
            for i, col in enumerate(self.columns):
                ax = axes[i]
                hist_data = self.original_data[col].iloc[-min(50, len(self.original_data)):]
                ax.plot(hist_data.index, hist_data.values, 'b-', label='Historical', linewidth=1.5)
                ax.plot(forecast_df_orig.index, forecast_df_orig[col], 'r-', label='Forecast', linewidth=2)
                ax.fill_between(forecast_df_orig.index, ci_lower_df_orig[col], ci_upper_df_orig[col],
                                alpha=0.3, color='red', label=f'{100 * (1 - self.ci_alpha)}% CI')
                ax.set_title(f'Forecast for {col} (VARIMA({self.best_p},{max(self.diff_orders.values())},{self.best_q}))')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            for j in range(n_vars, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.show()
        
        return {
            'point': forecast_df_orig,
            'ci_lower': ci_lower_df_orig,
            'ci_upper': ci_upper_df_orig
        }

    def simulate(self, n_periods: int = 100, plot: bool = True, tol: float = 1e-6) -> np.ndarray:
        """
        Simulate time series data from the fitted VARIMA model in the original scale.

        Parameters:
        -----------
        n_periods : int, optional
            Number of periods to simulate (default: 100).
        plot : bool, optional
            Whether to plot simulated series (default: True).
        tol : float, optional
            Tolerance for numerical stability (default: 1e-6).

        Returns:
        --------
        np.ndarray
            Simulated series of shape (n_periods, K) in the original scale.
        """
        # Simulate in differenced scale using parent VARMA simulate method
        Y_sim_diff = super().simulate(n_periods=n_periods, plot=False, tol=tol)
        
        K = len(self.columns)
        max_d = max(self.diff_orders.values())
        Y_sim_orig = np.zeros((n_periods, K))
        
        # Reverse differencing for each column
        for i, col in enumerate(self.columns):
            d = self.diff_orders.get(col, 0)
            last_values = self.original_data[col].values[-d:] if d > 0 else []
            is_log = self.I.is_log.get(col, False)
            
            # Reverse log transformation if applied
            sim_series = np.exp(Y_sim_diff[:, i]) if is_log else Y_sim_diff[:, i]
            
            # Reverse differencing
            if d == 0:
                Y_sim_orig[:, i] = sim_series
            else:
                cumsum = np.zeros(n_periods)
                for j in range(n_periods):
                    if j < d:
                        cumsum[j] = last_values[-d + j] if j < len(last_values) else 0
                    if j > 0:
                        cumsum[j] += cumsum[j - 1]
                    cumsum[j] += sim_series[j]
                Y_sim_orig[:, i] = cumsum
        
        # Create DataFrame for plotting
        T = len(self.original_data)
        sim_dates = pd.date_range(
            start=self.original_data.index[-1] + pd.Timedelta(days=1),
            periods=n_periods,
            freq=self.original_data.index.inferred_freq or 'D'
        ) if isinstance(self.original_data.index, pd.DatetimeIndex) else range(T, T + n_periods)
        sim_df = pd.DataFrame(Y_sim_orig, index=sim_dates, columns=self.columns)
        
        # Plotting
        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for i, col in enumerate(self.columns):
                axes[i].plot(sim_df.index, sim_df[col], label=f'Simulated {col}', color='r', linewidth=1.5)
                axes[i].set_title(f'Simulated Series for {col} (VARIMA({self.best_p},{max(self.diff_orders.values())},{self.best_q}))')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return Y_sim_orig