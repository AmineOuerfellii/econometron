import pandas as pd
import numpy as np
from scipy.stats import chi2, shapiro , jarque_bera
from statsmodels.tsa.stattools import acf, adfuller 
import matplotlib.pyplot as plt
from itertools import product
from econometron.utils.data_preparation import process_time_series
from econometron.utils.estimation.OLS import ols_estimator

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
    def plot_forecasts(self, h=None, figsize=(12, 8)):
        """
        Plot forecasts with confidence intervals for all variables.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting forecasts.")
        
        h = h or self.forecast_horizon
        forecasts = self.forecast(self.model_data.to_numpy(), self.best_model['beta'], self.best_p, h)
        
        # Create forecast dates
        if isinstance(self.model_data.index, pd.DatetimeIndex):
            forecast_dates = pd.date_range(
                start=self.model_data.index[-1] + pd.Timedelta(days=1), 
                periods=h, 
                freq=self.model_data.index.freq or 'D'
            )
        else:
            forecast_dates = range(len(self.model_data), len(self.model_data) + h)
        
        n_vars = len(self.columns)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_vars > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.columns):
            ax = axes[i]
            
            # Plot historical data (last 50 points for clarity)
            hist_data = self.model_data[col].iloc[-50:]
            ax.plot(hist_data.index, hist_data.values, 'b-', label='Historical', linewidth=1.5)
            
            # Plot forecasts
            forecast_values = forecasts['point'][:, i]
            ci_lower = forecasts['ci_lower'][:, i]
            ci_upper = forecasts['ci_upper'][:, i]
            
            ax.plot(forecast_dates, forecast_values, 'r-', label='Forecast', linewidth=2)
            ax.fill_between(forecast_dates, ci_lower, ci_upper, alpha=0.3, color='red', label='95% CI')
            
            ax.set_title(f'Forecast for {col}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(n_vars, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, figsize=(15, 10)):
        """
        Plot residuals analysis for all variables.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting residuals.")
        
        residuals = self.best_model['residuals']
        n_vars = len(self.columns)
        
        fig, axes = plt.subplots(n_vars, 3, figsize=figsize)
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(self.columns):
            resid = residuals[:, i]
            
            # Time series plot of residuals
            axes[i, 0].plot(resid, 'b-', linewidth=1)
            axes[i, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[i, 0].set_title(f'Residuals - {col}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Residual')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[i, 1].hist(resid, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i, 1].set_title(f'Residual Distribution - {col}')
            axes[i, 1].set_xlabel('Residual Value')
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(resid, dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f'Q-Q Plot - {col}')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_diagnostics(self, figsize=(15, 12)):
        """
        Plot comprehensive diagnostic plots including ACF, PACF, and statistical tests.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting diagnostics.")
        
        residuals = self.best_model['residuals']
        n_vars = len(self.columns)
        
        fig, axes = plt.subplots(n_vars, 4, figsize=figsize)
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(self.columns):
            resid = residuals[:, i]
            
            # ACF plot
            from statsmodels.tsa.stattools import acf
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            plot_acf(resid, lags=20, ax=axes[i, 0], title=f'ACF - {col}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # PACF plot
            plot_pacf(resid, lags=20, ax=axes[i, 1], title=f'PACF - {col}')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Squared residuals (for heteroscedasticity)
            axes[i, 2].plot(resid**2, 'g-', linewidth=1)
            axes[i, 2].set_title(f'Squared Residuals - {col}')
            axes[i, 2].set_xlabel('Time')
            axes[i, 2].set_ylabel('Squared Residual')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Rolling statistics
            window = min(20, len(resid)//4)
            if window > 1:
                rolling_mean = pd.Series(resid).rolling(window=window).mean()
                rolling_std = pd.Series(resid).rolling(window=window).std()
                
                axes[i, 3].plot(rolling_mean, label='Rolling Mean', linewidth=2)
                axes[i, 3].plot(rolling_std, label='Rolling Std', linewidth=2)
                axes[i, 3].axhline(y=0, color='r', linestyle='--', alpha=0.7)
                axes[i, 3].set_title(f'Rolling Statistics - {col}')
                axes[i, 3].set_xlabel('Time')
                axes[i, 3].set_ylabel('Value')
                axes[i, 3].legend()
                axes[i, 3].grid(True, alpha=0.3)
            else:
                axes[i, 3].text(0.5, 0.5, 'Insufficient data\nfor rolling stats', 
                            ha='center', va='center', transform=axes[i, 3].transAxes)
                axes[i, 3].set_title(f'Rolling Statistics - {col}')
        
        plt.tight_layout()
        plt.show()
        
        # Print diagnostic test results
        print("\n" + "="*60)
        print("DIAGNOSTIC TEST RESULTS")
        print("="*60)
        
        for col in self.columns:
            print(f"\n{col.upper()}:")
            print("-" * 40)
            
            if col in self.residual_diag_results:
                diag = self.residual_diag_results[col]
                
                print(f"Mean: {diag['mean']:.6f}")
                print(f"Variance: {diag['variance']:.6f}")
                
                # Ljung-Box test
                lb_stat = diag['ljung_box']['statistic']
                lb_pval = diag['ljung_box']['p_value']
                lb_result = "FAIL" if lb_pval < 0.05 else "PASS"
                print(f"Ljung-Box Test: Stat={lb_stat:.4f}, p-value={lb_pval:.4f} [{lb_result}]")
                
                # Shapiro-Wilk test
                sw_stat = diag['shapiro_wilk']['statistic']
                sw_pval = diag['shapiro_wilk']['p_value']
                sw_result = "FAIL" if sw_pval < 0.05 else "PASS"
                print(f"Shapiro-Wilk Test: Stat={sw_stat:.4f}, p-value={sw_pval:.4f} [{sw_result}]")
                
                # Additional test: Jarque-Bera
                resid = residuals[:, self.columns.index(col)]
                try:
                    jb_stat, jb_pval = jarque_bera(resid)
                    jb_result = "FAIL" if jb_pval < 0.05 else "PASS"
                    print(f"Jarque-Bera Test: Stat={jb_stat:.4f}, p-value={jb_pval:.4f} [{jb_result}]")
                except:
                    print("Jarque-Bera Test: Could not compute")
        
        print("\n" + "="*60)
        print("TEST INTERPRETATION:")
        print("PASS = Residuals satisfy the assumption (good)")
        print("FAIL = Residuals violate the assumption (potential issue)")
        print("Ljung-Box: Tests for autocorrelation (want p > 0.05)")
        print("Shapiro-Wilk: Tests for normality (want p > 0.05)")
        print("Jarque-Bera: Tests for normality (want p > 0.05)")
        print("="*60)