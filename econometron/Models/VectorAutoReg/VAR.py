import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from econometron.utils.estimation import ols_estimator
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson 
from scipy.stats import shapiro, norm ,jarque_bera ,probplot,multivariate_normal
import logging
from statsmodels.stats.diagnostic import acorr_ljungbox ,het_arch ,breaks_cusumolsresid
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class VAR_:
  def __init__(self,data,max_p=2,columns=None,criterion='AIC',forecast_horizon=10,plot=True,bootstrap_n=1000,ci_alpha=0.05,orth=False,check_stationarity=True,method=None,Threshold=0.8):
    self.data=data
    self.max_p=max_p
    self.criterion=criterion
    self.forecast_horizon=forecast_horizon
    self.plot=plot
    self.ci_alpha=ci_alpha
    self.check_stationarity=check_stationarity
    self.stationarity_results = {}
    self.thershold=Threshold
    ######
    self.coeff_table=pd.DataFrame()
    ###
    self.fitted=False
    self.best_model=None 
    self.best_p=None
    self.best_criterion_value=None
    self.all_results=[]
    self._validate_the_data(data)
    if method=="EL_RAPIDO":
      print("="*30,"Fitting the model","="*30)
      self.fit(columns)
      if self.fitted:
        print("the Model is well Fitted ...")
        print("="*30,"Predictions","="*30)
        self.predict(forecast_horizon, plot=plot, tol=1e-6)
        if bootstrap_n is not None:
          boots=True
        print("="*30,"Impulse Responses ","="*30)  
        self.impulse_res(h=forecast_horizon, orth=orth, bootstrap=boots, n_boot=bootstrap_n , plot=self.plot, tol=1e-6)
        print("="*30,"FEVD ","="*30)  
        self.FEVD(h=forecast_horizon, plot=plot)
        print("="*30,"Simulations ","="*30)  
        self.simulate(n_periods=100, plot=plot, tol=1e-6)
  #####################
  def _adf_test(self,series):
    try :
      if len(series.dropna())<2:
        raise ValueError("Series are Too short to apply an ADF test")
      results=adfuller(series.dropna(),autolag='AIC')
      return {'P_value':results[1], 'statistic': results[0], 'critical_values': results[4]}
    except Exception as e:
      logger.warning(f"ADF test failed: {e}")
      return {'P_value':1.0, 'statistic':np.nan, 'critical_values': {}}
  def _Kpss_test(self,series):
    try :
      if len(series.dropna())<2:
        raise ValueError("Series are Too short to apply an ADF test")
      results=kpss(series.dropna(),regression='c', nlags='auto')
      return {'P_value':results[1], 'statistic': results[0], 'critical_values': results[3]}
    except Exception as e:
      logger.warning(f"KPSS test failed: {e}")
      return {'P_value':1.0, 'statistic':np.nan, 'critical_values': {}}

  def _validate_the_data(self,data):
    ####Data type check
    if isinstance(data,pd.DataFrame) or isinstance(data,pd.Series):
      if isinstance(data,pd.Series):
        data=data.to_frame()
      pass
    else:
      raise ValueError("The input data must be a pandas DataFrame")
    lengths = [len(data[col]) for col in data.columns]
    if len(set(lengths)) > 1:
      raise ValueError("All columns must have the same length")
    #####check for Nan Values 
    if any(data[col].isna().any() for col in data.columns):
      raise ValueError("Columns is entirely or contains NaN values")
    #==================Stationarity validation====================#
    if self.check_stationarity:
      print("Performing stationarity checks...")
      for col in self.data.columns:
        series = self.data[col] 
        adf_result = self._adf_test(series)
        kpss_result = self._Kpss_test(series) # Corrected method name
        self.stationarity_results[col] = {'adf': adf_result,'kpss': kpss_result}
      for col in self.stationarity_results:
        print(f"\nColumn: {col}")
        verdict=True
        if self.stationarity_results[col]['adf']['P_value'] > 0.05 and self.stationarity_results[col]['kpss']['P_value'] < 0.05:
          verdict=False
          print(f"Verdict , The serie : {col} is not stationary")
        else:
          print(f"Verdict , The serie : {col} is stationary")
        self.stationarity_results[col]=verdict
        if np.all(list(self.stationarity_results.values())):
          self.data=data
        else:
          self.data=None
          raise ValueError("Data needs to be stationnary")
    else:
      print("Skipping stationarity checks - assuming data is stationary")

  #===================Getting to the Juicy part ==========================>>>>>>>

  ### First as we do we start by defining the lag Matrix 
  def lag_matrix(self,lags):
    data=self.data
    T,K=data.shape
    if T <= lags:
      raise ValueError("lags are superior to the series length")
    else:
      X=np.ones((T-lags,0))
      for lag in range(1,lags+1):
        lag_data=data[lags-lag:T-lag]
        if lag_data.ndim==1:
          lag_data=lag_data.reshape(-1,1)
        X=np.hstack((X,lag_data))
      Y=data[lags:]
      return X,Y
  #### Now let's compute aic and Bic :
  def _compute_aic_bic(self,Y,resids,K,P,T):
    resid_cov=np.cov(resids.T, bias=True)
    log_det=np.log(np.linalg.det(resid_cov + 1e-10 * np.eye(K)))
    n_params=K*(K*P+1)
    aic=T*log_det+2*n_params
    bic=T*log_det+n_params*np.log(T)
    return aic,bic
  ### the Fit method
  def fit(self,columns=None,p=None):
    ###First let's see an verify if data are now stationnary:
    s=1
    if np.all(list(self.stationarity_results.values())):
      pass 
    else:
      raise ValueError("Data needs to be stationnary")
    ### Now lets supose the user didn't enter any columns but the data contains some other types other numbers
    if columns is None:
      print("Selecting only columns with numeric data")
      columns = self.data.select_dtypes(include=np.number).columns.tolist()
    ###Then we check if the columns exits
    if len(columns) > len(self.data.columns):
      raise ValueError("the number of The columns doesn't match that of the input's data columns")
    if set(columns)!=set(self.data.columns):
      raise ValueError("Some of The Columns don't exist in your data input")
    ### Now the selection criterion
    if self.criterion not in ['AIC','BIC','aic','bic']:
      raise ValueError("The criterion must be either AIC or BIC")
    ###########
    self.columns=columns
    T, K = self.data.shape
    min_obs=self.max_p*K+1
    if T<min_obs:
      raise ValueError(f"Insufficient observations ({T}) for max_p={self.max_p} with {K} variables.")
    self.best_criterion_value = float('inf')
    self.all_results = []
    if p is not None and isinstance(p, int):
        s=p
        self.max_p=p
    for p in range(s,self.max_p+1):
      try:
        X,Y=self.lag_matrix(p)
        beta,fitted,resids,res=ols_estimator(X,Y)
        aic,bic=self._compute_aic_bic(Y,resids,K,p,T)
        self.all_results.append({
          'p':p,
          'beta':beta,
          'fitted':fitted,
          'residuals':resids,
          'fullresults':res,
          'aic':aic,
          'bic':bic
        })
        if self.criterion in ['AIC','aic']:
          if aic < self.best_criterion_value:
            self.best_criterion_value = aic
            self.best_model = self.all_results[-1]
            self.best_p = p
      except Exception as e:
        print(f'Failed for p={p}: {e}')
        continue
    if self.best_model is None:
      raise ValueError("No valid VAR model could be fitted") 
    C=len(self.columns)
    for i,col in enumerate(self.columns) :
      for lag in range(self.best_p):
        for j, var in enumerate(columns):
          idx=1+lag*C+j
          self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_coef'] = self.best_model['beta'][idx, i]
          self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_se'] = self.best_model['fullresults']['se'][idx, i]
          self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_z'] = self.best_model['fullresults']['z_values'][idx, i]
          self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_p'] = self.best_model['fullresults']['p_values'][idx, i]
      self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_coef'] = self.best_model['beta'][idx, i]
      self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_se'] = self.best_model['fullresults']['se'][idx, i]
      self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_z'] = self.best_model['fullresults']['z_values'][idx, i]
      self.coeff_table.loc[f'Lag_{lag+1}_{var}', f'{col}_p'] = self.best_model['fullresults']['p_values'][idx, i]
    ##### needs to complete all , add summary ,and plot option to see we've done on the fitting on train data or whatever
    ###
    if self.best_model:
      print(self.coeff_table)
      self.run_full_diagnosis(plot=self.plot,threshold=self.thershold)
      if self.plot==True:
        print("Plots are below")
        if not self.fitted:
           logger.warning("Model not fully fitted; predictions may be unreliable.")
        K = len(self.columns)
        p = self.best_model['p']
        fitted = self.best_model['fitted']
        train_data = self.data.iloc[p:] 
        if fitted.shape[0] != len(train_data):
            raise ValueError(f"Fitted values shape {fitted.shape} does not match training data length {len(train_data)}")
        fitted_df = pd.DataFrame(fitted, index=train_data.index, columns=self.columns)
        n_vars = K
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
        axes = np.array(axes).flatten() if n_vars > 1 else [axes]
        for i, col in enumerate(self.columns):
            ax = axes[i]
            ax.plot(train_data.index, train_data[col], 'b-', label='Original Train Data', linewidth=1.5)
            ax.plot(fitted_df.index, fitted_df[col], 'r--', label='VAR Fitted Values', linewidth=1.5)
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
  def run_full_diagnosis(self, num_lags=8,plot=False,threshold=0.8):
    if not 0 <= threshold <= 1:
      raise ValueError("Threshold needs to be between 0 and 1")
    Diagnosis = {}
    K = len(self.columns)  # Number of variables (columns in residuals)
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
        raise ValueError(f"Residuals have {resids.shape[1]} columns, expected {K}")
    # Warn if sample size is too small for Ljung-Box
    if resids.shape[0] < num_lags:
        print(f"Warning: Sample size ({resids.shape[0]}) < num_lags ({num_lags})")
    ###==================Serial Correlation===========================####
    print("===================Serial COrrelation Tests===========================")
    ## Using Durbin-Watson and Ljung-Box
    lb_results = []
    dw_results = []
    print(f"===============Ljung–Box Test (lags={num_lags})==============")
    for i in range(K):
        lb_test = acorr_ljungbox(resids[:, i], lags=[num_lags], return_df=True)
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
    ###Heteroscadisty
    print("==================Heteroscedasticity=================")
    Homoscedasicity=True
    arch_res=[]
    for i in range(K):
      arch_test = het_arch(resids[:, i])
      arch_res.append('pass' if arch_test[1] >=  0.5 else 'Fail')
      print(f"Residual {i}: ARCH p-value = {arch_test[1]:.4f} → {arch_res[i]}")
    arch_tests=arch_res.count('pass')/K
    if arch_tests != 1:
      Homoscedasicity=False
    Diagnosis['Heteroscedasticity_score']=arch_tests
    Diagnosis['Heteroscedasticity_diagnosis']='Passed' if arch_tests == 1 else 'Failed'
    Diagnosis['Final Heteroscedasticity Diagnosis'] = 'Passed' if Homoscedasicity else 'Failed'
    ###Normal_dist of residuals
    print("=======================Normality Test=======================")
    Normality = True
    jb_results = []
    shapiro_results = []
    for i in range(K):
        jb_test = jarque_bera(resids[:, i])
        sh_test = shapiro(resids[:, i])
        jb_pval = jb_test.pvalue
        sh_pval = sh_test.pvalue
        print(f"Residual {i}: JB p-value = {jb_pval:.4f}, Shapiro p-value = {sh_pval:.4f}")
        jb_results.append('pass' if jb_pval >= 0.05 else 'fail')
        shapiro_results.append('pass' if sh_pval >= 0.05 else 'fail')
    # Count passes (only count a variable if both tests passed)
    joint_passes = sum(1 for j, s in zip(jb_results, shapiro_results) if j == 'pass' and s == 'pass')
    normality_score = joint_passes / K
    if normality_score != 1:
        Normality = False
    Diagnosis['Normality_score'] = normality_score
    Diagnosis['Normality_diagnosis'] = 'Passed' if normality_score == 1 else 'Failed'
    Diagnosis['Final Normality Diagnosis'] = 'Passed' if Normality else 'Failed'
    ###Testing_for_structural_breaks
    print("#========================Structural Breaks============================")
    No_Structural_breaks=True
    cusum_stat, cusum_pval, _ = breaks_cusumolsresid(resids, ddof=0)
    print(f"CUSUM p-value : {cusum_pval:.4f}")
    #p > 0.05 for pass
    if cusum_pval < 0.05:
      No_Structural_breaks=False
    if No_Structural_breaks:
      print("No structural breaks detected")
    else:
      print("Structural breaks detected")
    Diagnosis['Final Structural Breaks']='Passed' if No_Structural_breaks else 'Failed'
    ################## Finish tests #############
    # Calculate final score as the average of test scores
    structural_breaks_score = 1.0 if No_Structural_breaks else 0.0
    final_score = (Diagnosis['DW_score'] + Diagnosis['LB_score'] + 
                   Diagnosis['Autocorrelation_score'] + 
                   Diagnosis['Heteroscedasticity_score'] + 
                   Diagnosis['Normality_score'] + structural_breaks_score) / 6
    # Assign verdict based on threshold
    self.fitted = final_score >= threshold
    Diagnosis['Final_score'] = final_score
    Diagnosis['Verdict'] = 'Passed' if self.fitted else 'Failed'
    # Create summary table
    print("\n==================Diagnostic Summary=================")
    summary_table = {
        'Estimation': 'OLS',
        'Model':f'VAR({self.best_p})',
        'Log-Likelihood': self.best_model.get('fullresults').get('log_likelihood','N/A'),
        'R-squared': self.best_model.get('fullresults').get('R2', 'N/A'),
        'AIC': self.best_model.get('aic', 'N/A'),
        'BIC': self.best_model.get('bic', 'N/A'),
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
      fig, axes = plt.subplots(nrows=K + 1, ncols=2, figsize=(12, fig_height))
      # === 1. Global CUSUM Plot (across both columns) ===
      flat_resid = resids.flatten()
      centered = flat_resid - np.mean(flat_resid)
      cusum = np.cumsum(centered)
      c = 0.948
      threshold = c * np.sqrt(len(cusum))
      # Merge the two top axes into one
      ax_cusum = plt.subplot2grid((K + 1, 2), (0, 0), colspan=2)
      ax_cusum.plot(cusum, label='Global CUSUM of Residuals')
      ax_cusum.axhline(y=threshold, color='red', linestyle='--', label='+95% Band')
      ax_cusum.axhline(y=-threshold, color='red', linestyle='--', label='-95% Band')
      ax_cusum.axhline(y=0, color='black', linestyle='-', label='Zero Line')
      ax_cusum.set_title("Global CUSUM Test (All Residuals)")
      ax_cusum.set_xlabel("Flattened Time Index")
      ax_cusum.set_ylabel("CUSUM Value")
      ax_cusum.legend()
      # === 2. Histogram + Q–Q Plots per residual ===
      for i in range(K):
          ax_hist = plt.subplot2grid((K + 1, 2), (i + 1, 0))
          ax_hist.hist(resids[:, i], bins=30, density=True, alpha=0.7, color='steelblue')
          ax_hist.set_title(f"Histogram of Residual {i}")
          ax_hist.set_xlabel("Residual Value")
          ax_hist.set_ylabel("Density")
          ax_qq = plt.subplot2grid((K + 1, 2), (i + 1, 1))
          probplot(resids[:, i], dist="norm", plot=ax_qq)
          ax_qq.set_title(f"Q–Q Plot for Residual {i}")
      plt.tight_layout()
      plt.show()
    return Diagnosis
  def _orthogonalize(self, Sigma):
      return np.linalg.cholesky(Sigma)
  def impulse_res(self, h=10, orth=True, bootstrap=False, n_boot=1000, plot=False, tol=1e-6):
      if self.best_model is None:
          raise ValueError("No model fitted. Cannot compute IRF.")
      K = len(self.columns)
      p = self.best_model['p']
      beta = self.best_model['beta']
      intercept_included = beta.shape[0] == K * p + 1
      A = beta[1:] if intercept_included else beta
      A = A.reshape(p, K, K).transpose(1, 2, 0)
      Psi = np.zeros((h, K, K))
      Psi[0] = np.eye(K)
      for i in range(1, h):
          for j in range(min(i, p)):
              Psi[i] += A[:, :, j] @ Psi[i - j - 1]
      if orth:
          Sigma = np.cov(self.best_model['residuals'].T)
          P = self._orthogonalize(Sigma)
          irf = np.array([Psi[i] @ P for i in range(h)])
      else:
          irf = Psi
      if not bootstrap:
          if plot:
              fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
              axes = axes.flatten() if K > 1 else [axes]
              for i in range(K):
                  for j in range(K):
                      idx = i * K + j
                      axes[idx].plot(range(h), irf[:, i, j], label=f'Shock {self.columns[j]} → {self.columns[i]}')
                      axes[idx].set_title(f'{self.columns[i]} response to {self.columns[j]} shock')
                      axes[idx].set_xlabel('Horizon')
                      axes[idx].set_ylabel('Response')
                      axes[idx].grid(True)
                      axes[idx].legend()
              plt.tight_layout()
              plt.show()
          return irf
      boot_irfs = np.zeros((n_boot, h, K, K))
      residuals = self.best_model['residuals']
      T, K = residuals.shape
      data = self.data.values
      for b in range(n_boot):
          boot_idx = np.random.choice(T, size=T, replace=True)
          boot_resids = residuals[boot_idx]
          Y_sim = np.zeros((T + p, K))
          Y_sim[:p] = data[-p:]
          intercept = beta[0] if intercept_included else np.zeros(K)
          for t in range(p, T + p):
              Y_t = intercept.copy()
              for j in range(p):
                  Y_t += A[:, :, j] @ Y_sim[t - j - 1]
              Y_t += boot_resids[t - p]
              Y_sim[t] = Y_t
          Y_sim = Y_sim[p:]
          X, Y = self.lag_matrix(p)
          try:
              boot_beta, _, _, _ = ols_estimator(X, Y_sim, tol=tol)
          except Exception as e:
              logger.warning(f"Bootstrap iteration {b} failed: {e}")
              continue
          boot_A = boot_beta[1:] if boot_beta.shape[0] == K * p + 1 else boot_beta
          boot_A = boot_A.reshape(p, K, K).transpose(1, 2, 0)
          boot_Psi = np.zeros((h, K, K))
          boot_Psi[0] = np.eye(K)
          for i in range(1, h):
              for j in range(min(i, p)):
                  boot_Psi[i] += boot_A[:, :, j] @ boot_Psi[i - j - 1]
          if orth:
              boot_Sigma = np.cov(boot_resids.T)
              try:
                  P = self._orthogonalize(boot_Sigma)
                  boot_irf = np.array([boot_Psi[i] @ P for i in range(h)])
              except np.linalg.LinAlgError:
                  logger.warning(f"Bootstrap iteration {b} failed: Non-positive definite covariance")
                  continue
          else:
              boot_irf = boot_Psi
          boot_irfs[b] = boot_irf
      ci_lower = np.percentile(boot_irfs, 100 * self.ci_alpha / 2, axis=0)
      ci_upper = np.percentile(boot_irfs, 100 * (1 - self.ci_alpha / 2), axis=0)
      if plot:
          fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
          axes = axes.flatten() if K > 1 else [axes]
          for i in range(K):
              for j in range(K):
                  idx = i * K + j
                  axes[idx].plot(range(h), irf[:, i, j], label=f'Shock {self.columns[j]} → {self.columns[i]}')
                  axes[idx].fill_between(range(h), ci_lower[:, i, j], ci_upper[:, i, j], 
                                          alpha=0.3, color='red', label=f'{100 * (1 - self.ci_alpha)}% CI')
                  axes[idx].set_title(f'{self.columns[i]} response to {self.columns[j]} shock')
                  axes[idx].set_xlabel('Horizon')
                  axes[idx].set_ylabel('Response')
                  axes[idx].grid(True)
                  axes[idx].legend()
          plt.tight_layout()
          plt.show()
      return irf, ci_lower, ci_upper

  def FEVD(self, h=10, plot=False):
      K = len(self.columns)
      irf = self.impulse_res(h=h, orth=True, bootstrap=False, plot=False)
      Sigma = np.cov(self.best_model['residuals'].T)
      fevd = np.zeros((h, K, K))
      mse = np.zeros((h, K))
      for i in range(h):
          for j in range(K):
              for t in range(i + 1):
                  mse[i, j] += np.sum(irf[t, j, :] ** 2 * np.diag(Sigma))
              for k in range(K):
                  fevd[i, j, k] = np.sum(irf[:i + 1, j, k] ** 2 * Sigma[k, k]) / mse[i, j] if mse[i, j] != 0 else 0
      if plot:
          fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
          axes = [axes] if K == 1 else axes
          for j in range(K):
              bottom = np.zeros(h)
              for k in range(K):
                  axes[j].bar(range(h), fevd[:, j, k], bottom=bottom, label=f'Shock from {self.columns[k]}')
                  bottom += fevd[:, j, k]
              axes[j].set_title(f'FEVD for {self.columns[j]}')
              axes[j].set_xlabel('Horizon')
              axes[j].set_ylabel('Variance Contribution')
              axes[j].legend()
              axes[j].grid(True, alpha=0.3)
          plt.tight_layout()
          plt.show()
      return fevd

  def simulate(self, n_periods=100, plot=False, tol=1e-6):
      K = len(self.columns)
      p = self.best_model['p']
      beta = self.best_model['beta']
      intercept_included = beta.shape[0] == K * p + 1
      intercept = beta[0] if intercept_included else np.zeros(K)
      A = beta[1:] if intercept_included else beta
      A = A.reshape(p, K, K).transpose(1, 2, 0)
      Sigma = np.cov(self.best_model['residuals'].T)
      Y_sim = np.zeros((n_periods + p, K))
      Y_sim[:p] = self.data.values[-p:]
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
      return Y_sim

  def predict(self, n_periods=1, plot=True, tol=1e-6):
      if self.best_model is None:
          raise ValueError("No model fitted. Cannot generate forecasts.")
      if not self.fitted:
          logger.warning("The model is not fully fitted; forecasts may be unreliable.")
      K = len(self.columns)
      p = self.best_model['p']
      beta = self.best_model['beta']
      intercept_included = beta.shape[0] == K * p + 1
      intercept = beta[0] if intercept_included else np.zeros(K)
      A = beta[1:] if intercept_included else beta
      A = A.reshape(p, K, K).transpose(1, 2, 0)
      h = n_periods if n_periods > 0 else self.forecast_horizon
      if isinstance(self.data.index, pd.DatetimeIndex):
          forecast_dates = pd.date_range(
              start=self.data.index[-1] + pd.Timedelta(days=1),
              periods=h,
              freq=self.data.index.freq or 'D'
          )
      else:
          forecast_dates = range(len(self.data), len(self.data) + h)
      forecasts = np.zeros((h, K))
      forecast_vars = np.zeros((h, K))
      last_observations = self.data.values[-p:].copy()
      Sigma = np.cov(self.best_model['residuals'].T)
      Psi = np.zeros((h, K, K))
      Psi[0] = np.eye(K)
      for i in range(1, h):
          for j in range(min(i, p)):
              Psi[i] += A[:, :, j] @ Psi[i - j - 1]
      for t in range(h):
          X_forecast = np.ones((1, 1)) if intercept_included else np.zeros((1, 0))
          for lag in range(1, p + 1):
              lag_data = forecasts[t - lag:t - lag + 1] if t >= lag else last_observations[p - lag:p - lag + 1]
              X_forecast = np.hstack((X_forecast, lag_data))
          forecasts[t] = X_forecast @ beta
          for j in range(K):
              var = 0
              for s in range(t + 1):
                  var += Psi[s, j, :] @ Sigma @ Psi[s, j, :].T
              forecast_vars[t, j] = var
      se = np.sqrt(forecast_vars)
      ci_lower = forecasts - norm.ppf(1 - self.ci_alpha / 2) * se
      ci_upper = forecasts + norm.ppf(1 - self.ci_alpha / 2) * se
      forecast_df = pd.DataFrame(forecasts, index=forecast_dates, columns=self.columns)
      ci_lower_df = pd.DataFrame(ci_lower, index=forecast_dates, columns=self.columns)
      ci_upper_df = pd.DataFrame(ci_upper, index=forecast_dates, columns=self.columns)
      if plot:
          n_vars = len(self.columns)
          n_cols = min(2, n_vars)
          n_rows = (n_vars + n_cols - 1) // n_cols
          fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
          axes = np.array(axes).flatten() if n_vars > 1 else [axes]
          for i, col in enumerate(self.columns):
              ax = axes[i]
              hist_data = self.data[col].iloc[-min(50, len(self.data)):]
              ax.plot(hist_data.index, hist_data.values, 'b-', label='Historical', linewidth=1.5)
              ax.plot(forecast_df.index, forecast_df[col], 'r-', label='Forecast', linewidth=2)
              ax.fill_between(forecast_df.index, ci_lower_df[col], ci_upper_df[col],
                              alpha=0.3, color='red', label=f'{100 * (1 - self.ci_alpha)}% CI')
              ax.set_title(f'Forecast for {col}')
              ax.set_xlabel('Time')
              ax.set_ylabel('Value')
              ax.legend()
              ax.grid(True, alpha=0.3)
          for j in range(n_vars, len(axes)):
              axes[j].set_visible(False)
          plt.tight_layout()
          plt.show()
      return {'point': forecast_df, 'ci_lower': ci_lower_df, 'ci_upper': ci_upper_df}