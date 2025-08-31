import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from econometron.utils.estimation.Regression import ols_estimator
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, norm, jarque_bera, probplot, multivariate_normal
import logging
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, breaks_cusumolsresid
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


warnings.simplefilter("ignore", InterpolationWarning)


class VAR:
    def __init__(self, data, max_p=2, columns=None, criterion='AIC', forecast_horizon=10, plot=True, bootstrap_n=1000, ci_alpha=0.05, orth=False, check_stationarity=True, Key=None, Threshold=0.8 ,verbose=False):
        self.data = data
        self.max_p = max_p
        self.criterion = criterion
        self.forecast_horizon = forecast_horizon
        self.plot = plot
        if ci_alpha < 1 and ci_alpha > 0:
            self.ci_alpha = ci_alpha
        else:
            self.ci_alpha = 0.05
            print('ci_alpha must be between 0 and 1 fallback to defaul : 0.05')
        self.check_stationarity = check_stationarity
        self.stationarity_results = {}
        self.thershold = Threshold if Threshold < 1 and Threshold > 0 else 0.8
        ######
        self.coeff_table = pd.DataFrame()
        ###
        self.fitted = False
        self.best_model = None
        self.best_p = None
        self.best_criterion_value = None
        self.all_results = []
        self.roots = []
        self._validate_the_data(data, columns=columns, verbose=verbose)
        if Key == "EL_RAPIDO":
            print("="*30, "Fitting the model", "="*30)
            self.fit(columns)
            if self.fitted:
                print("the Model is well Fitted ...")
                print("="*30, "Predictions", "="*30)
                self.predict(forecast_horizon, plot=plot)
                if bootstrap_n is not None:
                    boots = True
                print("="*30, "Impulse Responses ", "="*30)
                self.impulse_res(h=forecast_horizon, orth=orth, bootstrap=boots,
                                 n_boot=bootstrap_n, plot=self.plot, tol=1e-6)
                print("="*30, "FEVD ", "="*30)
                self.FEVD(h=forecast_horizon, plot=plot)
                print("="*30, "Simulations ", "="*30)
                self.simulate(n_periods=100, plot=plot, tol=1e-6)
        elif Key == 'SbS':
            print("="*15, "Fitting the model based on the model initialization", "="*15)
            self.fit(columns)

            def choice(output=True):
                user_input = input("Enter your criterion (AIC, BIC, HQIC): ")
                if user_input.lower() in ['aic', 'bic', 'hqic']:
                    self.criterion = user_input.upper()
                    logger.info(
                        "="*15 + f" Order selection sorted by: {self.criterion} " + "="*15)
                    table = self.fit(columns, get_order=True)
                    if output:
                        print(table)
                    return table
                else:
                    logger.warning(
                        "Invalid criterion. Please choose AIC, BIC, or HQIC.")
                    return choice()

            def fit_var(table, o=None):
                try:
                    if o is None:
                        o = int(input("Select VAR order (p): "))
                    if o in table.index:
                        self.fit(p=o)
                        print("Model refitted with p =", o)
                    else:
                        logger.warning("Selected order not found in table.")
                except Exception as e:
                    logger.warning("Error during VAR fitting: " + str(e))

            def fitting_var():
                user_input = input("Do you want to refit the model? (y/n): ")
                if user_input.lower() == 'y':
                    user_input = input(
                        "Do you want to change the criterion? (y/n): ")
                    if user_input.lower() == 'y':
                        tab = choice()
                        fit_var(tab)
                    else:
                        tab = self.fit(columns, get_order=True)
                        print(tab)
                        fit_var(tab)
            # Step-by-step interactive fitting
            fitting_var()
            if self.fitted:
                logger.info("The model is fitted")
                response = input(
                    "Do you want to compute Impulse Responses? (y/n): ")
                if response.lower() == "y":
                    print("="*30, "Impulse Responses", "="*30)
                    self.impulse_res(h=forecast_horizon, orth=orth,
                                     bootstrap=bootstrap_n is not None, n_boot=bootstrap_n, plot=plot, tol=1e-6)
                response = input(
                    "Do you want to compute FEVD (Forecast Error Variance Decomposition)? ((y/n): ")
                if response.lower() == "y":
                    print("="*30, "FEVD", "="*30)
                    self.FEVD(h=forecast_horizon, plot=plot)
                response = input("Do you want to generate Forecasts? (y/n): ")
                if response.lower() == "y":
                    print("="*30, "Forecast", "="*30)
                    self.predict(forecast_horizon, plot=plot, tol=1e-6)
                response = input("Do you want to run a Simulation? (y/n): ")
                if response.lower() == "y":
                    print("="*30, "Simulation", "="*30)
                    self.simulate(n_periods=100, plot=plot, tol=1e-6)

    #####################
    def _adf_test(self, series):
        try:
            series_clean = series.dropna()
            n = len(series_clean)
            if n < 5:
                raise ValueError("Series too short to apply ADF test")
            max_lag = min(int(12 * (n / 100) ** 0.25), n - 2)
            results = adfuller(series_clean, maxlag=max_lag, autolag='AIC')
            return {'test_statistic': results[0],'p_value': results[1],'used_lag': results[2],'nobs': results[3],'critical_values': results[4],'icbest': results[5]
            }
        except Exception as e:
            logger.warning(f"ADF test failed for series '{series.name}': {e}")
            return {
                'test_statistic': np.nan,
                'p_value': 1.0,
                'used_lag': None,
                'nobs': None,
                'critical_values': {},
                'icbest': None
            }


    def _Kpss_test(self, series):
        try:
            series_clean = series.dropna()
            n = len(series_clean)
            if n < 5:
                raise ValueError("Series too short to apply KPSS test")
            nlags = min(int(np.floor(12 * (n / 100) ** 0.25)), n - 2)
            results = kpss(series_clean, regression='c', nlags=nlags)
            return {
                'test_statistic': results[0],
                'p_value': results[1],
                'lags_used': results[2],
                'critical_values': results[3]
            }
        except Exception as e:
            logger.warning(f"KPSS test failed for series '{series.name}': {e}")
            return {
                'test_statistic': np.nan,
                'p_value': 1.0,
                'lags_used': None,
                'critical_values': {}
            }

    def _validate_the_data(self, data, columns=None, verbose: bool = False):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame or Series")
        if columns is not None:
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"The following columns are not in the data: {missing_cols}")
            data = data[columns]
        numeric_cols = data.select_dtypes(include=[np.integer, np.floating]).columns
        non_numeric_cols = set(data.columns) - set(numeric_cols)
        if non_numeric_cols and verbose:
            logging.info(f"Skipping columns with non-numeric dtypes (not int or float): {non_numeric_cols}")
        if not numeric_cols.all():
            raise ValueError("No numeric columns found in the data")
        data = data[numeric_cols]
        lengths = [len(data[col]) for col in data.columns]
        if len(set(lengths)) > 1:
            raise ValueError("All columns must have the same length")
        if any(data[col].isna().any() for col in data.columns):
            raise ValueError("Columns contain NaN values")

        # ================== Stationarity validation =====================#
        if self.check_stationarity:
            if verbose:
                logging.info("Performing stationarity checks on numeric columns...")
            self.stationarity_results = {}
            summary_table = []
            for col in data.columns:
                series = data[col]
                if series.nunique() <= 1:
                    if verbose:
                        logging.info(f"Skipping stationarity test for '{col}': constant or near-constant series")
                    self.stationarity_results[col] = {
                        'adf': {'test_statistic': np.nan, 'p_value': np.nan, 'used_lag': 0},
                        'kpss': {'test_statistic': np.nan, 'p_value': np.nan, 'lags_used': 0},
                        'stationary': False
                    }
                    continue

                adf_result = self._adf_test(series)
                kpss_result = self._Kpss_test(series)
                stationary = adf_result['p_value'] <= 0.05 or kpss_result['p_value'] >= 0.05
                self.stationarity_results[col] = {
                    'adf': adf_result,
                    'kpss': kpss_result,
                    'stationary': stationary
                }
                if verbose:
                    summary_table.append({
                        'Column': col,
                        'ADF Stat': round(adf_result['test_statistic'], 4) if not np.isnan(adf_result['test_statistic']) else np.nan,
                        'ADF p-value': round(adf_result['p_value'], 4) if not np.isnan(adf_result['p_value']) else np.nan,
                        'ADF Lags': adf_result['used_lag'],
                        'KPSS Stat': round(kpss_result['test_statistic'], 4) if not np.isnan(kpss_result['test_statistic']) else np.nan,
                        'KPSS p-value': round(kpss_result['p_value'], 4) if not np.isnan(kpss_result['p_value']) else np.nan,
                        'KPSS Lags': kpss_result['lags_used'],
                        'Stationary': stationary
                    })

            if verbose and summary_table:
                df_summary = pd.DataFrame(summary_table)
                logging.info("\nStationarity Test Summary:\n" + df_summary.to_string(index=False))
            non_stationary_cols = [col for col, val in self.stationarity_results.items() if not val['stationary']]
            if non_stationary_cols:
                if len(non_stationary_cols) == len(data.columns):
                    if verbose:
                        logging.info(f"All columns are non-stationary: {non_stationary_cols}")
                    raise ValueError(f"All columns are non-stationary: {non_stationary_cols}")
                if verbose:
                    logging.info(f"Non-stationary columns removed: {non_stationary_cols}")
                self.data = data.drop(columns=non_stationary_cols, errors='ignore')
            else:
                self.data = data
        else:
            if verbose:
                logging.info("Skipping stationarity checks - assuming data is stationary")
            self.data = data

        return self.data

    # ===================Getting to the Juicy part ==========================>>>>>>>
    # First as we do we start by defining the lag Matrix

    def lag_matrix(self, lags):
        data = self.data
        T, K = data.shape
        if T <= lags:
            raise ValueError("lags are superior to the series length")
        else:
            X = np.ones((T-lags, 0))
            for lag in range(1, lags+1):
                lag_data = data[lags-lag:T-lag]
                if lag_data.ndim == 1:
                    lag_data = lag_data.reshape(-1, 1)
                X = np.hstack((X, lag_data))
            Y = data[lags:]
            return X, Y
    # Now let's compute aic and Bic :

    def _compute_aic_bic_hqic(self, resids, K, P, T):
        resid_cov = np.cov(resids.T, bias=True)
        # print('resids_cov',resid_cov)
        log_det = np.log(np.linalg.det(resid_cov + 1e-10 * np.eye(K)))
        n_params = K*(K*P+1)
        # print(n_params)
        hqic = log_det+2*np.log(np.log(T))*n_params/T
        aic = log_det+(2*n_params)/T
        bic = log_det+n_params*np.log(T)/T
        return aic, bic, hqic
    # Build coeff table :

    def build_and_display_coeff_table(self):
        if self.best_model is None:
            print("No model fitted. Cannot build coefficient table.")
            return
        beta = self.best_model['beta']
        se = self.best_model['fullresults']['se']
        z_values = self.best_model['fullresults']['z_values']
        p_values = self.best_model['fullresults']['p_values']
        var_names = self.columns if hasattr(self, 'columns') else [
            f"Var{i+1}" for i in range(len(self.data.columns))]
        K = len(var_names)
        if not hasattr(self, 'coeff_table') or self.coeff_table is None:
            columns = [f"{col}_{stat}" for col in var_names for stat in [
                'coef', 'se', 'z', 'p']]
            index = [
                'Constant'] + [f'Lag_{lag+1}_{var}' for lag in range(self.best_p) for var in var_names]
            self.coeff_table = pd.DataFrame(index=index, columns=columns)
        for i, col in enumerate(var_names):
            self.coeff_table.loc['Constant', f'{col}_coef'] = beta[0, i]
            self.coeff_table.loc['Constant',
                                 f'{col}_se'] = se[0, i] if se.shape[0] > 0 else 0
            self.coeff_table.loc['Constant',
                                 f'{col}_z'] = z_values[0, i] if z_values.shape[0] > 0 else 0
            self.coeff_table.loc['Constant',
                                 f'{col}_p'] = p_values[0, i] if p_values.shape[0] > 0 else 0

        for lag in range(self.best_p):
            for j, var in enumerate(var_names):
                row_idx = 1 + lag * K + j
                row_name = f'Lag_{lag+1}_{var}'
                if row_idx < beta.shape[0]:
                    for i, col in enumerate(var_names):
                        self.coeff_table.loc[row_name,
                                             f'{col}_coef'] = beta[row_idx, i]
                        self.coeff_table.loc[row_name,
                                             f'{col}_se'] = se[row_idx, i] if row_idx < se.shape[0] else 0
                        self.coeff_table.loc[row_name, f'{col}_z'] = z_values[row_idx,
                                                                              i] if row_idx < z_values.shape[0] else 0
                        self.coeff_table.loc[row_name, f'{col}_p'] = p_values[row_idx,
                                                                              i] if row_idx < p_values.shape[0] else 0

        print("=" * 120)
        print(f"VAR({self.best_p}) Coefficient Table")
        print("=" * 120)

        print("\nConstant Parameters:")
        print("-" * 120)
        print(f"{'Variable':<15}", end="")
        for col in var_names:
            print(
                f"| {col+'_coef':<12} {col+'_se':<10} {col+'_z':<8} {col+'_p':<8}", end="")
        print()
        print("-" * 120)
        print(f"{'Constant':<15}", end="")
        for col in var_names:
            coef = self.coeff_table.loc['Constant', f'{col}_coef']
            se_val = self.coeff_table.loc['Constant', f'{col}_se']
            z_val = self.coeff_table.loc['Constant', f'{col}_z']
            p_val = self.coeff_table.loc['Constant', f'{col}_p']
            print(
                f"| {coef:>10.4f} {se_val:>10.4f} {z_val:>8.4f} {p_val:>8.4f}", end="")
        print()

        for lag in range(1, self.best_p + 1):
            print(f"\nLag {lag} Parameters:")
            print("-" * 120)
            print(f"{'Variable':<15}", end="")
            for col in var_names:
                print(
                    f"| {col+'_coef':<12} {col+'_se':<10} {col+'_z':<8} {col+'_p':<8}", end="")
            print()
            print("-" * 120)
            for var in var_names:
                row_name = f'Lag_{lag}_{var}'
                if row_name in self.coeff_table.index:
                    print(f"{var:<15}", end="")
                    for col in var_names:
                        coef = self.coeff_table.loc[row_name, f'{col}_coef']
                        se_val = self.coeff_table.loc[row_name, f'{col}_se']
                        z_val = self.coeff_table.loc[row_name, f'{col}_z']
                        p_val = self.coeff_table.loc[row_name, f'{col}_p']
                        print(
                            f"| {coef:>10.4f} {se_val:>10.4f} {z_val:>8.4f} {p_val:>8.4f}", end="")
                    print()

        print("=" * 120)
    # order_select:
    def select_order(self):
        self.all_results = []
        select_order_table = None
        T, K = self.data.shape
        for p in range(1, self.max_p+1):
            X, Y = self.lag_matrix(p)
            beta, fitted, resids, res = ols_estimator(X, Y)
            aic, bic, hqic = self._compute_aic_bic_hqic(resids, K, p, T)
            self.all_results.append({
                'p': p,
                'beta': beta,
                'fitted': fitted,
                'residuals': resids,
                'fullresults': res,
                'aic': aic,
                'bic': bic,
                'hqic': hqic
            })
        criterion = self.criterion.lower()
        if criterion in ['aic', 'bic', 'hqic']:
            select_order_table = pd.DataFrame(self.all_results)[['p', 'aic', 'bic', 'hqic']]
            select_order_table['AIC*'] = ''
            select_order_table['BIC*'] = ''
            select_order_table['HQIC*'] = ''
            min_aic_idx = select_order_table['aic'].idxmin()
            min_bic_idx = select_order_table['bic'].idxmin()
            min_hqic_idx = select_order_table['hqic'].idxmin()        
            select_order_table.loc[min_aic_idx, 'AIC*'] = '*'
            select_order_table.loc[min_bic_idx, 'BIC*'] = '**'
            select_order_table.loc[min_hqic_idx, 'HQIC*'] = '***'
            ranks = select_order_table[['aic', 'bic', 'hqic']].rank(method='min')
            select_order_table['Avg_Rank'] = ranks.mean(axis=1)
            select_order_table['Best_Model'] = ''
            best_model_idx = select_order_table['Avg_Rank'].idxmin()
            select_order_table.loc[best_model_idx, 'Best_Model'] = 'Best'
            select_order_table = select_order_table.sort_values(by=criterion).reset_index(drop=True)
        return select_order_table
        # this test , so for now i'll keep it this way , although we can use this in the next func fit but for now let's keep it this way
    # the Fit method

    def fit(self, columns=None, p=None, output=True, get_order=False):
        # First let's see an verify if data are now stationnary:
        if np.all(list(self.stationarity_results.values())):
            pass
        else:
            raise ValueError("Data needs to be stationnary")
        # Now lets supose the user didn't enter any columns but the data contains some other types other numbers
        if columns is None:
            print("Selecting only columns with numeric data")
            columns = self.data.select_dtypes(
                include=np.number).columns.tolist()
        # Then we check if the columns exits
        if len(columns) > len(self.data.columns):
            raise ValueError(
                "the number of The columns doesn't match that of the input's data columns")
        if set(columns) != set(self.data.columns):
            raise ValueError(
                "Some of The Columns don't exist in your data input")
        # Now the selection criterion
        if self.criterion not in ['AIC', 'BIC', 'HQIC', 'aic', 'bic', 'hqic']:
            raise ValueError("The criterion must be either AIC,BIC or HQIC")
        ###########
        self.columns = columns
        T, K = self.data.shape
        min_obs = self.max_p*K+1
        if T < min_obs:
            raise ValueError(
                f"Insufficient observations ({T}) for max_p={self.max_p} with {K} variables.")
        self.best_criterion_value = float('inf')
        self.all_results = []
        s = 1
        if p:
            s = p
            self.max_p = p
        # print(s,self.max_p)
        for p in range(s, self.max_p+1):
            try:
                X, Y = self.lag_matrix(p)
                beta, fitted, resids, res = ols_estimator(X, Y)
                aic, bic, hqic = self._compute_aic_bic_hqic(resids, K, p, T)
                self.all_results.append({
                    'p': p,
                    'beta': beta,
                    'fitted': fitted,
                    'residuals': resids,
                    'fullresults': res,
                    'aic': aic,
                    'bic': bic,
                    'hqic': hqic
                })
                criterion = self.criterion.lower()
                if criterion in ['aic', 'bic', 'hqic']:
                    if get_order:
                        select_order_table = pd.DataFrame(self.all_results)[
                            ['p', 'aic', 'bic', 'hqic']].sort_values(by=criterion).reset_index(drop=True)
                        # print(select_order_table)
                        return select_order_table
                    crit = locals()[self.criterion.lower()]
                    if crit < self.best_criterion_value:
                        # print(self.best_criterion_value)
                        self.best_criterion_value = crit
                        self.best_model = self.all_results[-1]
                        self.best_p = p
            except Exception as e:
                print(f'Failed for p={p}: {e}')
                continue
        if self.best_model is None:
            raise ValueError("No valid VAR model could be fitted")

        # C=len(self.columns)
        # for i,col in enumerate(self.columns) :
        #   for lag in range(self.best_p):
        #     for j, var in enumerate(columns):
        #       idx=1+lag*C+j
        #       self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_coef'] = self.best_model['beta'][idx, i]
        #       self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_se'] = self.best_model['fullresults']['se'][idx, i]
        #       self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_z'] = self.best_model['fullresults']['z_values'][idx, i]
        #       self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_p'] = self.best_model['fullresults']['p_values'][idx, i]
        #   self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_coef'] = self.best_model['beta'][idx, i]
        #   self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_se'] = self.best_model['fullresults']['se'][idx, i]
        #   self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_z'] = self.best_model['fullresults']['z_values'][idx, i]
        #   self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_p'] = self.best_model['fullresults']['p_values'][idx, i]
        # needs to complete all , add summary ,and plot option to see we've done on the fitting on train data or whatever
        # Note :completed that below
        if self.best_model:
            if output:
                self.build_and_display_coeff_table()
                self.run_full_diagnosis(
                    plot=self.plot, threshold=self.thershold)
                if self.plot == True:
                    print("Plots are below")
                    if not self.fitted:
                        logger.warning(
                            "Model not fully fitted; predictions may be unreliable.")
                    K = len(self.columns)
                    p = self.best_model['p']
                    fitted = self.best_model['fitted']
                    train_data = self.data.iloc[p:]
                    if fitted.shape[0] != len(train_data):
                        raise ValueError(
                            f"Fitted values shape {fitted.shape} does not match training data length {len(train_data)}")
                    fitted_df = pd.DataFrame(
                        fitted, index=train_data.index, columns=self.columns)
                    n_vars = K
                    n_cols = min(2, n_vars)
                    n_rows = (n_vars + n_cols - 1) // n_cols
                    fig, axes = plt.subplots(
                        n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
                    axes = np.array(axes).flatten() if n_vars > 1 else [axes]
                    for i, col in enumerate(self.columns):
                        ax = axes[i]
                        ax.plot(
                            train_data.index, train_data[col], 'b-', label='Original Train Data', linewidth=1.5)
                        ax.plot(
                            fitted_df.index, fitted_df[col], 'r--', label='VAR Fitted Values', linewidth=1.5)
                        ax.set_title(f'{col}: Original vs Fitted')
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Value')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    for j in range(n_vars, len(axes)):
                        axes[j].set_visible(False)
                    plt.tight_layout()
                    plt.show()
            return self.best_model
        else:
            raise ValueError("No valid VAR model")
         ########

    def _companion_matrix(self):
        beta = self.best_model['beta']
        K = len(self.columns)
        p = self.best_model['p']
        intercept_included = beta.shape[0] == K * p + 1
        A = beta[1:] if intercept_included else beta
        try:
            A = A.reshape(p, K, K).transpose(0, 2, 1)
        except ValueError as e:
            raise ValueError(
                f"Cannot reshape beta into ({p}, {K}, {K}). Beta shape: {beta.shape}, A shape: {A.shape}")
        cm = np.zeros((K * p, K * p))
        for i in range(p):
            cm[:K, i * K:(i + 1) * K] = A[i]
        if p > 1:
            cm[K:, :-K] = np.eye(K * (p - 1))
        eigvals = np.linalg.eigvals(cm)
        stable = np.all(np.abs(eigvals) < 1 - 1e-6)
        return cm, stable, eigvals

    def run_full_diagnosis(self, num_lags=8, plot=False, threshold=0.8):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold needs to be between 0 and 1")
        Diagnosis = {}
        self.columns = self.data.columns
        # Number of variables (columns in residuals)
        K = len(self.data.columns)
        # Check if model is fitted
        if self.best_model is None:
            print("No model fitted. Cannot perform diagnostics.")
            Diagnosis['Final Diagnosis'] = 'Not Fitted'
            return Diagnosis
        # Get residuals
        results = self.best_model['fullresults']
        resids = results['resid']
        # Validate residuals shape
        if resids.shape[1] != K:
            raise ValueError(
                f"Residuals have {resids.shape[1]} columns, expected {K}")
        # Warn if sample size is too small for Ljung-Box
        if resids.shape[0] < num_lags:
            print(
                f"Warning: Sample size ({resids.shape[0]}) < num_lags ({num_lags})")
        ## =====================Stability=================================####
        print("===================Stability===========================")
        cm, stable, eigs = self._companion_matrix()
        print(f"The VAR model is stable:{stable:.4f}")
        Diagnosis['Final Stability Diagnosis'] = 'Stable' if stable else "Not Stable"
        S_score = 2 if stable else 0
        Diagnosis['Stability Score'] = S_score
        ### ==================Serial Correlation===========================####
        print("===================Serial COrrelation Tests===========================")
        # Using Durbin-Watson and Ljung-Box
        lb_results = []
        dw_results = []
        print(f"===============Ljung–Box Test (lags={num_lags})==============")
        for i in range(K):
            lb_test = acorr_ljungbox(
                resids[:, i], lags=[num_lags], return_df=True)
            pval = lb_test['lb_pvalue'].values[0]
            LbT = "PASS" if pval > 0.05 else "FAIL"
            print(f"Residual {i}: p-value = {pval:.4f} → {LbT}")
            lb_results.append(LbT)
        print("==================Durbin-Watson Statistics=================")
        for i in range(K):
            dw = durbin_watson(resids[:, i])
            dw_result = "Pass" if 1.5 <= dw <= 2.5 else "Fail"
            print(f"Residual {i}: DW = {dw:.4f} → {dw_result}")
            dw_results.append(dw_result)
        # Calculate scores
        DW_score = dw_results.count('Pass') / K
        LB_score = lb_results.count('PASS') / K
        # Calculate autocorrelation score (average number of tests passed per residual)
        auto_corr_score = 0
        for dw_res, lb_res in zip(dw_results, lb_results):
            tests_passed = 0
            if dw_res == "Pass":
                tests_passed += 1
            if lb_res == "PASS":
                tests_passed += 1
            auto_corr_score += tests_passed / 2  # Each residual contributes 0, 0.5, or 1
        auto_corr_score /= K  # Average over all residuals
        # Populate Diagnosis dictionary
        Diagnosis['DW_score'] = DW_score
        Diagnosis['LB_score'] = LB_score
        Diagnosis['Autocorrelation_score'] = auto_corr_score
        Diagnosis['DW_diagnosis'] = 'Passed' if DW_score == 1 else 'Failed'
        Diagnosis['LB_diagnosis'] = 'Passed' if LB_score == 1 else 'Failed'
        Diagnosis['Autocorrelation_diagnosis'] = 'Passed' if auto_corr_score == 1 else 'Failed'
        Diagnosis['Final auocorrelation Diagnosis'] = 'Passed' if DW_score == 1 and LB_score == 1 else 'Failed'
        # Heteroscadisty
        print("==================Heteroscedasticity=================")
        Homoscedasicity = True
        arch_res = []
        for i in range(K):
            arch_test = het_arch(resids[:, i])
            arch_res.append('pass' if arch_test[1] >= 0.05 else 'Fail')
            print(
                f"Residual {i}: ARCH p-value = {arch_test[1]:.4f} → {arch_res[i]}")
        arch_tests = arch_res.count('pass')/K
        if arch_tests != 1:
            Homoscedasicity = False
        Diagnosis['Heteroscedasticity_score'] = arch_tests
        Diagnosis['Heteroscedasticity_diagnosis'] = 'Passed' if arch_tests == 1 else 'Failed'
        Diagnosis['Final Heteroscedasticity Diagnosis'] = 'Passed' if Homoscedasicity else 'Failed'
        # Normal_dist of residuals
        print("=======================Normality Test=======================")
        Normality = True
        jb_results = []
        shapiro_results = []
        for i in range(K):
            jb_test = jarque_bera(resids[:, i])
            sh_test = shapiro(resids[:, i])
            jb_pval = jb_test.pvalue
            sh_pval = sh_test.pvalue
            print(
                f"Residual {i}: JB p-value = {jb_pval:.4f}, Shapiro p-value = {sh_pval:.4f}")
            jb_results.append('pass' if jb_pval >= 0.05 else 'fail')
            shapiro_results.append('pass' if sh_pval >= 0.05 else 'fail')
        # Count passes (only count a variable if both tests passed)
        joint_passes = sum(1 for j, s in zip(
            jb_results, shapiro_results) if j == 'pass' and s == 'pass')
        normality_score = joint_passes / K
        if normality_score != 1:
            Normality = False
        Diagnosis['Normality_score'] = normality_score
        Diagnosis['Normality_diagnosis'] = 'Passed' if normality_score == 1 else 'Failed'
        Diagnosis['Final Normality Diagnosis'] = 'Passed' if Normality else 'Failed'
        # Testing_for_structural_breaks
        print("#========================Structural Breaks============================")
        No_Structural_breaks = True
        cusum_stat, cusum_pval, _ = breaks_cusumolsresid(resids, ddof=0)
        print(f"CUSUM p-value : {cusum_pval:.4f}")
        # p>0.05 for pass
        if cusum_pval < 0.05:
            No_Structural_breaks = False
        if No_Structural_breaks:
            print("No structural breaks detected")
        else:
            print("Structural breaks detected")
        Diagnosis['Final Structural Breaks'] = 'Passed' if No_Structural_breaks else 'Failed'
        ################## Finish tests #############
        # Calculate final score as the average of test scores
        structural_breaks_score = 1.0 if No_Structural_breaks else 0.0
        final_score = (Diagnosis['DW_score'] + Diagnosis['LB_score'] +
                       Diagnosis['Autocorrelation_score'] +
                       Diagnosis['Heteroscedasticity_score'] +
                       Diagnosis['Normality_score'] + structural_breaks_score+S_score) / 7
        # Assign verdict based on threshold
        self.fitted = final_score >= threshold
        Diagnosis['Final_score'] = final_score
        Diagnosis['Verdict'] = 'Passed' if self.fitted else 'Failed'
        # Create summary table
        print("\n==================Diagnostic Summary=================")
        summary_table = {
            'Estimation': 'OLS',
            'Model': f'VAR({self.best_p})',
            'Log-Likelihood': self.best_model.get('fullresults').get('log_likelihood', 'N/A'),
            'R-squared': self.best_model.get('fullresults').get('R2', 'N/A'),
            'AIC': self.best_model.get('aic', 'N/A'),
            'BIC': self.best_model.get('bic', 'N/A'),
            'Stability': f"{Diagnosis['Stability Score']:.4f}({Diagnosis['Final Stability Diagnosis']})",
            'DW Score': f"{Diagnosis['DW_score']:.4f} ({Diagnosis['DW_diagnosis']})",
            'LB Score': f"{Diagnosis['LB_score']:.4f} ({Diagnosis['LB_diagnosis']})",
            'Autocorrelation Score': f"{Diagnosis['Autocorrelation_score']:.4f} ({Diagnosis['Autocorrelation_diagnosis']})",
            'Heteroscedasticity Score': f"{Diagnosis['Heteroscedasticity_score']:.4f} ({Diagnosis['Heteroscedasticity_diagnosis']})",
            'Normality Score': f"{Diagnosis['Normality_score']:.4f} ({Diagnosis['Normality_diagnosis']})",
            'Structural Breaks': f"{structural_breaks_score:.4f} ({Diagnosis['Final Structural Breaks']})",
            'Final Score': f"{final_score:.4f}",
            'Verdict': Diagnosis['Verdict']
        }
        # Print table
        print("Model Diagnostics Summary:")
        print("-" * 50)
        for key, value in summary_table.items():
            print(f"{key:<30} | {value}")
        print("-" * 50)
        # Plotting section
        if plot:
            T, K = resids.shape
            fig_height = 4 * (K + 1)
            fig, axes = plt.subplots(
                nrows=K + 1, ncols=2, figsize=(12, fig_height))
            # === CUSUM Plot  ===
            cusum_stat, cusum_pval, cusum_crit = breaks_cusumolsresid(
                resids, ddof=0)
            flat_resid = resids.flatten()
            n = len(flat_resid)
            resid_centered = flat_resid - np.mean(flat_resid)
            resid_std = np.std(flat_resid, ddof=0)
            cusum_series = np.cumsum(resid_centered) / (resid_std * np.sqrt(n))
            critical_value = None
            for sig_level, crit_val in cusum_crit:
                if sig_level == 5:
                    critical_value = crit_val
                    break

            if critical_value is None:
                critical_value = 1.36
            ax_cusum = plt.subplot2grid((K + 1, 2), (0, 0), colspan=2)
            ax_cusum.plot(
                cusum_series, label='Standardized CUSUM of Residuals', color='blue')
            ax_cusum.axhline(y=critical_value, color='red', linestyle='--',
                             label=f'+Critical Value ({critical_value})')
            ax_cusum.axhline(y=-critical_value, color='red', linestyle='--',
                             label=f'-Critical Value ({-critical_value})')
            ax_cusum.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            test_result = "PASSED" if cusum_pval >= 0.05 else "FAILED"
            ax_cusum.set_title(
                f"CUSUM Test for Structural Breaks (p-value: {cusum_pval:.4f}, {test_result})")
            ax_cusum.set_xlabel("Time Index")
            ax_cusum.set_ylabel("Standardized CUSUM")
            ax_cusum.legend()
            ax_cusum.grid(True, alpha=0.3)
            # === 2. Histogram + Q–Q Plots per residual===
            for i in range(K):
                ax_hist = plt.subplot2grid((K + 1, 2), (i + 1, 0))
                ax_hist.hist(resids[:, i], bins=30,
                             density=True, alpha=0.7, color='steelblue')
                ax_hist.set_title(
                    f"Histogram of Residual {i} ({self.columns[i]})")
                ax_hist.set_xlabel("Residual Value")
                ax_hist.set_ylabel("Density")
                ax_qq = plt.subplot2grid((K + 1, 2), (i + 1, 1))
                probplot(resids[:, i], dist="norm", plot=ax_qq)
                ax_qq.set_title(
                    f"Q–Q Plot for Residual {i} ({self.columns[i]})")
            plt.tight_layout()
            plt.show()
        return Diagnosis

    def _orthogonalize(self, Sigma):
        return np.linalg.cholesky(Sigma)

    def predict(self, n_periods=1, plot=True):
        if self.best_model is None:
            raise ValueError("No model fitted. Cannot generate forecasts.")
        if not self.fitted:
            print("Warning: The model is not fully fitted; forecasts may be unreliable.")
        # === Model setup ===
        K = len(self.columns)
        p = self.best_model['p']
        beta = self.best_model['beta']
        Sigma = np.cov(self.best_model['residuals'].T)
        intercept = beta[0, :]
        coeffs = beta[1:, :]
        # print(coeffs.shape)
        # A = np.zeros((K, K, p))
        # for j in range(p):
        #     A[:, :, j] = coeffs[K*j:K*(j+1),:].reshape(K, K).T
        A = coeffs.reshape(p, K, K).transpose(2, 1, 0)
        # print(A.shape)
        # print(A[:,:,1])
        # === Forecast generation ===
        if isinstance(self.data.index, pd.DatetimeIndex):
            forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(
                days=1), periods=n_periods, freq=self.data.index.freq or 'D')
        else:
            forecast_dates = range(len(self.data), len(self.data)+n_periods)
        forecasts = np.zeros((n_periods, K))
        forecast_vars = np.zeros((n_periods, K))
        last_observations = self.data.values[-p:].copy()
        # print(last_observations)
        z_vector = np.flipud(last_observations).T
        # print(z_vector)
        Psi = np.zeros((n_periods, K, K))
        Psi[0] = np.eye(K)
        for s in range(1, n_periods):
            for j in range(1, min(p, s)+1):
                Psi[s] += Psi[s-j]@A[:, :, j-1]
        # print(Psi)
        for t in range(n_periods):
            forecast_t = intercept.copy()
            # print('inter',intercept)
            for lag in range(1, p+1):
                lag_idx = t-lag
                lag_val = forecasts[lag_idx] if lag_idx >= 0 else last_observations[lag_idx]
                # print('lag_val',lag_val)
                # print('a',A[:,:,lag-1])
                forecast_t += A[:, :, lag-1]@lag_val.T
                # print(forecast_t)
            forecasts[t] = forecast_t
            for j in range(K):
                var_j = 0
                for s in range(t+1):
                    psi = Psi[s, j, :]
                    var_j += psi@Sigma@psi.T
                forecast_vars[t, j] = max(var_j, 0)
        # print(forecast_t)
        se = np.sqrt(forecast_vars)
        z = norm.ppf(1-self.ci_alpha/2)
        ci_lower = forecasts-z*se
        ci_upper = forecasts+z*se
        forecast_df = pd.DataFrame(
            forecasts, index=forecast_dates, columns=self.columns)
        ci_lower_df = pd.DataFrame(
            ci_lower, index=forecast_dates, columns=self.columns)
        ci_upper_df = pd.DataFrame(
            ci_upper, index=forecast_dates, columns=self.columns)
        # print(forecast_df)
        if plot:
            T, _ = self.data.shape
            whole_data = pd.concat([self.data, forecast_df], axis=0)
            fig, axes = plt.subplots(len(self.columns), 1, figsize=(
                12, 4 * len(self.columns)), sharex=True)
            if len(self.columns) == 1:
                axes = [axes]
            for i, col in enumerate(self.columns):
                ax = axes[i]
                total_time = T + n_periods
                time_range = np.arange(total_time)
                ax.plot(time_range[:T], whole_data[col].iloc[:T],
                        'b-', label='Historical', linewidth=1.5)
                ax.plot(time_range[T-1:T+n_periods], whole_data[col].iloc[T-1:T +
                        n_periods], 'k--', linewidth=1.5, label='Forecast' if i == 0 else "")
                ax.fill_between(time_range[T:],
                                ci_lower_df[col].iloc[:n_periods],
                                ci_upper_df[col].iloc[:n_periods],
                                color='skyblue',
                                alpha=0.4,
                                label=f'{100 * (1 - self.ci_alpha):.0f}% CI' if i == 0 else "")
                ax.axvline(T - 1, color='gray', linestyle=':', linewidth=1)
                ax.set_title(f'Forecast for {col}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
            plt.tight_layout()
            plt.show()
        return {'point': forecast_df, 'ci_lower': ci_lower_df, 'ci_upper': ci_upper_df}
            # predict : ver

    def simulate(self, n_periods=100, plot=True, tol=1e-6):
        K = len(self.columns)
        p = self.best_model['p']
        beta = self.best_model['beta']
        intercept_included = beta.shape[0] == K * p + 1
        intercept = beta[0] if intercept_included else np.zeros(K)
        A = beta[1:] if intercept_included else beta
        A = A.reshape(p, K, K).transpose(2, 1, 0)
        # print(A[:,:,0])
        Sigma = np.cov(self.best_model['residuals'].T)
        Y_sim = np.zeros((n_periods+p, K))
        Y_sim[:p] = np.flipud(self.data.values[-p:])
        # Y_sim[:p] = self.data.values[-p:]
        # print(Y_sim)
        # print(Y_sim)
        for t in range(p, n_periods + p):
            Y_t = intercept.copy()
            for j in range(p):
                Y_t += A[:, :, j] @ Y_sim[t - j - 1]
                Y_t += multivariate_normal.rvs(mean=np.zeros(K), cov=Sigma)
                Y_sim[t] = Y_t
        Y_sim = Y_sim[p:]
        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for i in range(K):
                axes[i].plot(Y_sim[:, i], label=f'Simulated {self.columns[i]}')
                axes[i].set_title(f'Simulated Series for {self.columns[i]}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True)
            plt.tight_layout()
            plt.show()
        return {'simulations': Y_sim}
        # sim_ver

    def FEVD(self, h=10, plot=False):
        K = len(self.columns)
        irf = self.impulse_res(h=h, orth=True, bootstrap=False, plot=False)
        Sigma = np.cov(self.best_model['residuals'].T)
        P = np.linalg.cholesky(Sigma).T
        fevd = np.zeros((h+1, K, K))
        mse = np.zeros((h+1, K))
        for i in range(h+1):
            for j in range(K):
                for t in range(i+1):
                    phi_t = irf[t, j, :]
                    mse[i, j] += np.dot(phi_t, phi_t)
                for k in range(K):
                    fevd[i, j, k] = np.sum(
                        irf[:i+1, j, k] ** 2) / mse[i, j] if mse[i, j] != 0 else 0
        for i in range(h+1):
            for j in range(K):
                total = np.sum(fevd[i, j, :])
                if total > 0:
                    fevd[i, j, :] /= total
        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for j in range(K):
                bottom = np.zeros(h+1)
                for k in range(K):
                    axes[j].bar(range(h+1), fevd[:, j, k], bottom=bottom,
                                label=f'Shock from {self.columns[k]}')
                    bottom += fevd[:, j, k]
                axes[j].set_title(f'FEVD for {self.columns[j]}')
                axes[j].set_xlabel('Horizon')
                axes[j].set_ylabel('Variance Contribution')
                axes[j].legend()
                axes[j].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return fevd

    def impulse_res(self, h=10, orth=True, bootstrap=False, n_boot=1000, plot=False, tol=1e-6):
        if self.best_model is None:
            raise ValueError("No model fitted. Cannot compute IRF.")
        K = len(self.columns)
        p = self.best_model['p']
        beta = self.best_model['beta']
        intercept_included = beta.shape[0] == K * p + 1
        A = beta[1:] if intercept_included else beta
        A = A.reshape(p, K, K).transpose(2, 1, 0)
        # print(A)
        Psi = np.zeros((h+1, K, K))
        Psi[0] = np.eye(K)
        for i in range(1, h+1):
            for j in range(min(i, p)):
                Psi[i] += A[:, :, j] @ Psi[i-j-1]
            if orth:
                Sigma = np.cov(self.best_model['residuals'].T)
                P = self._orthogonalize(Sigma)
                irf = np.array([Psi[i] @ P for i in range(h+1)])
            else:
                irf = Psi
        # print(irf.shape)
        # z = 1.96
        # band_width = 0.05
        # ci_lower = irf - z * band_width
        # ci_upper = irf + z * band_width
        if not bootstrap:
            if plot:
                fig, axes = plt.subplots(K, K, figsize=(15, 15), sharex=True)
                axes = axes.flatten() if K > 1 else [axes]
                for i in range(K):
                    for j in range(K):
                        idx = i * K + j
                        axes[idx].plot(
                            range(h+1), irf[:, i, j], label=f'Shock {self.columns[j]} → {self.columns[i]}')
                        # axes[idx].fill_between(range(h+1), ci_lower[:, i, j], ci_upper[:, i, j],alpha=0.3, color='r', label=f'{100 * (1 - self.ci_alpha)}% CI')
                        axes[idx].set_title(
                            f'{self.columns[i]} response to {self.columns[j]} shock')
                        axes[idx].set_xlabel('Horizon')
                        axes[idx].set_ylabel('Response')
                        axes[idx].grid(True)
                        axes[idx].legend()
                plt.tight_layout()
                plt.show()
            return irf
        else:
            boot_irfs = np.zeros((n_boot, h+1, K, K))
            residuals = self.best_model['residuals']
            T, K = residuals.shape
            data = self.data.values
            for b in range(n_boot):
                boot_idx = np.random.choice(T, size=T, replace=True)
                boot_resids = residuals[boot_idx]
                Y_sim = np.zeros((T+p, K))
                Y_sim[:p] = np.flipud(data[-p:])
                intercept = beta[0] if intercept_included else np.zeros(K)
                for t in range(p, T+p):
                    Y_t = intercept.copy()
                    for j in range(p):
                        Y_t += A[:, :, j] @ Y_sim[t-j-1]
                    Y_t += boot_resids[t-p]
                    Y_sim[t] = Y_t
                Y_sim = Y_sim[p:]
                X, Y = self.lag_matrix(p)
                try:
                    boot_beta, _, _, _ = ols_estimator(X, Y_sim, tol=tol)
                except Exception as e:
                    logger.warning(f"Bootstrap iteration {b} failed: {e}")
                    continue
                boot_A = boot_beta[1:] if boot_beta.shape[0] == K * \
                    p + 1 else boot_beta
                boot_A = boot_A.reshape(p, K, K).transpose(2, 1, 0)
                boot_Psi = np.zeros((h+1, K, K))
                boot_Psi[0] = np.eye(K)
                for i in range(1, h+1):
                    for j in range(min(i, p)):
                        boot_Psi[i] += boot_A[:, :, j]@boot_Psi[i - j - 1]
                if orth:
                    boot_Sigma = np.cov(boot_resids.T)
                    try:
                        P = self._orthogonalize(boot_Sigma)
                        boot_irf = np.array(
                            [boot_Psi[i] @ P for i in range(h+1)])
                    except np.linalg.LinAlgError:
                        logger.warning(
                            f"Bootstrap iteration {b} failed: Non-positive definite covariance")
                        continue
                else:
                    boot_irf = boot_Psi
                boot_irfs[b] = boot_irf
            ci_lower = np.percentile(
                boot_irfs, 100 * self.ci_alpha / 2, axis=0)
            ci_upper = np.percentile(
                boot_irfs, 100 * (1 - self.ci_alpha / 2), axis=0)
            if plot:
                fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
                axes = axes.flatten() if K > 1 else [axes]
                for i in range(K):
                    for j in range(K):
                        idx = i * K + j
                        axes[idx].plot(
                            range(h+1), irf[:, i, j], label=f'Shock {self.columns[j]} → {self.columns[i]}')
                        axes[idx].fill_between(range(h+1), ci_upper[:, i, j], ci_lower[:, i, j],
                                               alpha=0.3, color='red', label=f'{100 * (1 - self.ci_alpha)}% CI')
                        axes[idx].set_title(
                            f'{self.columns[i]} response to {self.columns[j]} shock')
                        axes[idx].set_xlabel('Horizon')
                        axes[idx].set_ylabel('Response')
                        axes[idx].grid(True)
                        axes[idx].legend()
                plt.tight_layout()
                plt.show()
        return {'irf': irf, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
####Adding granger causality test or leavind it foe users ' nu we need full work 
