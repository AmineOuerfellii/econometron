import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.stattools import durbin_watson
from sklearn.cross_decomposition import CCA
from .VAR import VAR
from econometron.utils.estimation.OLS import ols_estimator
from econometron.utils.optimizers import minimize_qn
from scipy.stats import chi2, norm ,jarque_bera, shapiro,probplot,multivariate_normal
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch,breaks_cusumolsresid
from numpy.linalg import inv,eigvals,det,cholesky
from joblib import Parallel, delayed
import warnings
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VARMA_(VAR):
  def __init__(self,data,max_p=5,max_q=5,columns=None,forecast_h=6,plot=True,check_stationarity=True,bootstrap_n=1000,criterion='AIC',structural_id=False,ci_alpha=0.05,Key=None,Threshold=0.8,orth=False):
    super().__init__(data,max_p,columns,criterion,forecast_h,plot,bootstrap_n,ci_alpha,orth,check_stationarity,Key,Threshold)
    self.max_q=max_q
    self.structural_id=structural_id
    self.best_model=None
    self.fitted=False
    self.columns=self.data.columns
    self.key=Key
    #####
    self.Kronind=None
    self.best_model=None
    self.best_q=None
    self.best_p=None
    self.AR_s=None
    self.MA_s=None
    self.coeff_table=pd.DataFrame()
    self.Threshold=Threshold
    self.forecast_h=forecast_h
  ########### 
  def kron_index(self,lag):
      T,K=self.data.shape
      start_f=T-lag
      past=self.data.iloc[:start_f,:].to_numpy()
      p_r_ts=np.zeros((start_f,K*lag))
      p_r_ts[:,K*(lag-1):]=past
      for i in range(1,lag):
          p_i = self.data.iloc[i:i + start_f, :].to_numpy()
          p_r_ts[:, K * (lag - (i + 1)):K * (lag - i)] = p_i
      kdx=np.zeros(K, dtype=int)
      found=np.zeros(K, dtype=int)
      cstar=[]
      star=lag
      h=0
      futu=None
      while sum(found) < K:
          past=p_r_ts[:T-lag-h,:]
          #print(past.shape)
          futu1=self.data.iloc[star + h:T, :].to_numpy()
          #print(futu1.shape)
          if futu is not None and futu.shape[0] > past.shape[0]:
              futu=futu[:past.shape[0], :]
          for i in range(K):
              if found[i]==0:
                  if h==0:
                      s1=[j for j in range(K) if found[j] == 0 and j < i]+[i]
                      futu=futu1[:, s1]
                  else:
                      futu=np.column_stack((futu, futu1[:,i])) if futu is not None else futu1[:,i:i+1]
                  n=min(past.shape[1],futu.shape[1])
                  cca=CCA(n_components=n, scale=True)
                  X_c,Y_c=cca.fit_transform(past, futu)
                  corr=[np.corrcoef(X_c[:, j], Y_c[:, j])[0, 1] for j in range(X_c.shape[1])]
                  dp,df=past.shape[1], futu.shape[1]
                  #print(df)
                  deg = dp - df + 1
                  if h == 0:
                      dsq = 1
                  else:
                      x1 = X_c[:, n - 1]
                      y1 = Y_c[:, n - 1]
                      acfy = acf(y1, nlags=h, fft=False)[1:h + 1]
                      acfx = acf(x1, nlags=h, fft=False)[1:h + 1]
                      dsq = 1 + 2 * np.sum(acfx * acfy)
                  sccs = min(corr) ** 2
                  n = T-1
                  tst=-(n-0.5*(dp+df-1))*np.log(1-sccs/dsq)
                  pv=1-chi2.cdf(tst, deg)
                  stat=[tst,deg,pv]+([dsq] if h>0 else [])
                  cstar.append(stat)
                  print(f"Component {i + 1}: sccs={sccs:.6f}, tst={tst:.3f}, deg={deg}, pv={pv:.3f}, dsq={dsq:.3f}")
                  if pv > self.ci_alpha:
                      found[i]=1
                      kdx[i]=h
                      print(f"Component {i + 1}: Kronecker index {h}, pv={pv:.3f}")
                      if h > 0:
                          futu=futu[:, :df-1]
          h+=1
      return {"index": kdx,"tests": cstar}
  #########identify#####################
  def struct_id(self,ord=None,use_var=False,output=True):
    T=self.data.shape[0]
    order=0
    if ord is None and not use_var or use_var:
      logger.warning("Using VAR to identify structural parameters")
      if self.max_p >1 and self.max_p > T/3:
        var=super().fit(p=self.max_p,output=False)
        order=var['p']
      else:
        p=int(np.round(T/4)) ### Arbitrary choice
        var=super().fit(p=p,output=False)
        order=var['p']
    else:
      if ord and not use_var:
        order=ord
    kd=self.Kronind=self.kron_index(order)['index']
    K=len(kd)
    idx=np.argsort(kd)
    self.best_p=np.max(kd)
    self.best_q=self.best_p
    mx=(self.best_p+1)*K
    MA=np.full((K, mx),2,dtype=int)
    for i in range(K):
      MA[i,i]=1
      if kd[i]<self.best_q:
        j=(kd[i]+1)*K
        MA[i,j:mx]=0
    if K>1:
      for i in range(K-1):
        MA[i,i+1:K]=0
    AR=MA.copy()
    if K>1:
      for i in range(1,K):
        for j in range(i):
          if kd[j] <= kd[i]:
            AR[i,j]=0
    MA[:,:K]=AR[:,:K]
    for i in range(K):
      for j in range(K):
        if kd[i] > kd[j]:
          for n in range(1,kd[i]-kd[j]+1):
            AR[i,(n*K)+j]=0
    if output:
      print("AR coefficient matrix:")
      print(AR)
      print("MA coefficient matrix:")
      print(MA)
    return{"AR_s_id":AR,"MA_s_id":MA}
  ########################################
  def _ini_s1(self, ord=None, output=True, p=None, q=None):
      T, K = self.data.shape
      estims = []
      s_e = []
      max_q_p = int(round(T / 3))
      if ord is None:
          if self.max_p > 1 or self.max_q > 1:
              max_q_p = max(self.max_p, self.max_q) + 7 if max(self.max_p, self.max_q) + 7 < T else int(round(T / 3))
      else:
          if ord > 1:
              max_q_p = ord
      # Step 1: get residuals from high order VAR
      Hov = VAR(data=self.data, max_p=max_q_p).fit(output=False)
      resids = Hov['residuals']
      p_v = Hov['p']
      _, Y = super().lag_matrix(p_v)
      X = np.array(resids)   
      Y = np.array(Y)
      T1,K1=Y.shape
      # print("w-----shapes------w")
      # print("this data", Y.shape)
      # print("this res", X.shape)
      if self.structural_id:
          struct_matx = self.struct_id(ord=p_v, output=output)
          AR = self.AR_s = struct_matx['AR_s_id']
          MA = self.MA_s = struct_matx['MA_s_id']
          p_q = max(int(np.floor(AR.shape[1] / K) - 1), 1)
          for i in range(K):
              X_ = []
              Y_i = Y[p_q:T1,i]
              n_obs = Y_i.shape[0]
              # print('----- T',T1)
              # print('----- nobs',n_obs)
              if i > 0:
                  for j in range(i):
                      if AR[i, j] > 1:
                          tmp = X[p_q:T1,j]-Y[p_q:T1,j]
                          # print('if i>0',tmp.shape)
                          X_.append(tmp)
              for l in range(1,p_q+1):
                  j_ = l * K
                  for j in range(K):
                      idx = j_ + j
                      if AR[i, idx] > 1:
                        # print(p_q-l)
                        # print(T-l)
                        # print(T)
                        # print(T-l-p_q+l)
                        tmp=Y[p_q-l:T1-l,j]
                        #print('AR lag',tmp.shape)
                        X_.append(tmp)
              for ll in range(1,p_q+1):
                  j_=ll*K
                  for j in range(K):
                      idx=j_+j
                      if MA[i, idx] > 1:
                          tmp = X[p_q-ll:T1-ll, j]
                          #print('MA lag',tmp.shape)
                          X_.append(tmp)                   
              # print("w---", i, "------w")
              # print("Y_i shape:", Y_i.shape)
              # for x in X_:
              #   print(x.shape)
              # new_w=np.hstack(X_).reshape(-1,len(X_))
              # print(new_w.shape[0]==Y_i.shape[0])  
              # print("new_w shape:", new_w.shape)
              if X_:
                Y_i=Y_i.reshape(-1,1)
                # print("Y_i shape:", Y_i.shape)
                new_w=np.hstack(X_).reshape(-1,len(X_))
                beta_a, _, _, diag = ols_estimator(new_w, Y_i)
                estims.extend(beta_a)
                s_e.extend(diag['se'])
              else:
                  print(f'[Warning] No regressors found for variable {i}')
      else:
          # Fallback: no structural id
          ist = max(p or 0, q or 0)
          Y_ = Y[ist:, :]
          X_ = []
          # print(Y_.shape)
          # print(X.shape)
          # print(p)
          if p:
            # print('ok',p)
            for j in range(1,p+1):
              tmp = Y[ist-j:T1-j, :]
              #print(tmp)
              X_.append(tmp)
          if q:
            for j in range(1,q+1):
              tmp = X[ist-j:T1-j, :]
              #print(tmp)
              X_.append(tmp)
          if X_:
              X_combined = np.hstack(X_)
              Y_reshaped = Y_.reshape(-1, K)
              beta_a, _, _, diag = ols_estimator(X_combined, Y_reshaped)
              estims = beta_a
              s_e = diag['se']
          else:
              print('No regressors available.')
      return estims, s_e
    ###############
  def _prepare_for_est(self,estimates,stand_err,output=True):
    #print('eeee',estimates)
    if estimates is None or stand_err is None:
      raise ValueError('estimates are None , cannot proceed with estimation')
    par = np.array([])
    separ = np.array([])
    if len(estimates)==0:
      return par, separ,par,separ
    if not self.structural_id:
      est_vec = estimates.flatten(order='F')
      se_vec = stand_err.flatten(order='F')
      fixed = np.ones_like(estimates)
      mask = fixed.flatten(order='F') == 1
    else:
      beta_flat = np.concatenate([b.flatten() for b in estimates]) if isinstance(estimates, list) else estimates.flatten()
      se_flat = np.concatenate([s.flatten() for s in stand_err]) if isinstance(stand_err, list) else stand_err.flatten()
      est_vec = beta_flat
      se_vec = se_flat
      fixed = np.ones(len(est_vec), dtype=int)  # All parameters should be estimated
      mask = fixed == 1
    par = est_vec[mask]
    separ = se_vec[mask]
    # Calculate bounds
    lowerBounds = par - 2 * separ
    upperBounds = par + 2 * separ
    if output: 
      # print('est_vec:', est_vec)
      # print('se_vec:', se_vec)
      # print('mask:', mask)
      print(f"Number of parameters: {len(par)}")
      print(f"Initial estimates: {np.round(par, 4)}")
      print(f"Lower bounds: {np.round(lowerBounds, 4)}")
      print(f"Upper bounds: {np.round(upperBounds, 4)}")
    return par, separ, lowerBounds, upperBounds
    ############################## Likelihood 
  def log_likelihood(self, par, LB, UB, structural_id=False, PhiID=None, ThetaID=None, 
                    Kron_index=None, p=None, q=None, enforce_sta_inver=False):
      """
      Properly scaled log-likelihood function for BFGS optimization
      
      Key fixes:
      1. Realistic penalty scaling 
      2. Proper negative log-likelihood return
      3. Bounded penalties that don't overwhelm the likelihood
      4. Debug output to track what's happening
      """
      T, K = self.data.shape
      data = np.array(self.data)
      maxk = Kron_index.max() if structural_id else max(p, q)
      maxk = min(maxk, T - 1)
      nT = T
      par = np.array(par)
      
      # Initialize all penalties
      boundary_penalty = 0
      stability_penalty_ = 0
      numerical_penalty = 0
      
      # MUCH smaller penalty scaling - don't overwhelm the likelihood!
      penalty_scale = 100  # Was 1e6 - way too big!
      
      # Gentle boundary penalties
      eps = 1e-4
      for i, (p_val, lb, ub) in enumerate(zip(par, LB, UB)):
          if p_val < lb + eps:
              boundary_penalty += penalty_scale * (lb + eps - p_val)**2
          elif p_val > ub - eps:
              boundary_penalty += penalty_scale * (p_val - (ub - eps))**2
      
      # Gentle penalty for extreme parameter values
      extreme_threshold = 50  # Much more reasonable
      for p_val in par:
          if abs(p_val) > extreme_threshold:
              boundary_penalty += penalty_scale * ((abs(p_val) - extreme_threshold) / extreme_threshold)**2
      
      # Parameter extraction (keeping your structural_id logic)
      if structural_id:
          Kpar = par.copy()
          Ph0 = np.eye(K)
          kp1 = PhiID.shape[1]
          kp = kp1 - K
          PH = np.zeros((K, kp))
          Kidx = ThetaID.shape[0]
          TH = np.zeros((Kidx, kp))
          ARid = PhiID
          MAid = ThetaID
          icnt = 0

          for i in range(K):
              all_ar_idx = np.where(ARid[i, :] > 1)[0]
              all_ma_idx = np.where(MAid[i, :] > 1)[0]
              kdx = all_ar_idx[all_ar_idx < K]
              idx = all_ar_idx[all_ar_idx >= K]
              jdx = all_ma_idx[all_ma_idx >= K]

              if len(kdx) > 0:
                  Ph0[i, kdx] = Kpar[icnt:icnt + len(kdx)]
                  icnt += len(kdx)

              if len(idx) > 0:
                  col_positions = idx - K
                  valid_cols = (col_positions >= 0) & (col_positions < kp)
                  valid_idx = idx[valid_cols]
                  valid_positions = col_positions[valid_cols]
                  PH[i, valid_positions] = Kpar[icnt:icnt + len(valid_idx)]
                  icnt += len(valid_idx)

              if len(jdx) > 0:
                  col_positions = jdx - K
                  valid_cols = (col_positions >= 0) & (col_positions < kp)
                  valid_jdx = jdx[valid_cols]
                  valid_positions = col_positions[valid_cols]
                  TH[i, valid_positions] = Kpar[icnt:icnt + len(valid_jdx)]
                  icnt += len(valid_jdx)

          # Minimal regularization
          Ph0 += 1e-12 * np.eye(K)
          
          # Reasonable conditioning penalty
          try:
              cond_num = np.linalg.cond(Ph0)
              if cond_num > 1e6:  # More reasonable threshold
                  boundary_penalty += penalty_scale * np.log(cond_num / 1e6)
              Ph0i = inv(Ph0)
          except (np.linalg.LinAlgError, ValueError):
              # Hard return for genuine numerical failures
              return 1e10

          ARc = Ph0i @ PH
          MAc = Ph0i @ TH

      else:
          kp = K**2 * p if p is not None else 0
          kq = K**2 * q if q is not None else 0
          ARc = par[:kp].reshape(K, p*K, order='F') if p > 0 else np.zeros((K, 0))
          MAc = par[kp:kp + kq].reshape(K, q * K, order='F') if q > 0 else np.zeros((K, 0))

      # Reasonable stability penalties
      def stability_penalty(coef_matrix, max_lag):
          """Reasonably scaled stability penalty"""
          penalty = 0
          if coef_matrix.shape[1] > 0 and max_lag > 0:
              try:
                  companion = np.zeros((K * max_lag, K * max_lag))
                  if max_lag > 1:
                      companion[K:, :-K] = np.eye(K * (max_lag - 1))
                  for j in range(max_lag):
                      companion[:K, j * K:(j + 1) * K] = coef_matrix[:, j * K:(j + 1) * K]
                  
                  eigvals_comp = eigvals(companion)
                  if np.any(np.isnan(eigvals_comp)) or np.any(np.isinf(eigvals_comp)):
                      return 1e8  # Hard penalty for numerical breakdown
                  
                  eigvals_abs = np.abs(eigvals_comp)
                  max_eigval = np.max(eigvals_abs)
                  
                  # Gentle penalty approaching unit root
                  if max_eigval > 0.95:
                      # Quadratic penalty, not exponential!
                      penalty += penalty_scale * 10 * (max_eigval - 0.95)**2
                  if max_eigval >= 1.0:
                      # Harder penalty at unit root
                      penalty += penalty_scale * 100 * (max_eigval - 1.0)**2
                      
              except (np.linalg.LinAlgError, ValueError):
                  return 1e8
          return penalty
      
      if enforce_sta_inver:
          ar_lags = ARc.shape[1] // K if ARc.shape[1] > 0 else 0
          ma_lags = MAc.shape[1] // K if MAc.shape[1] > 0 else 0
          
          stability_penalty_ += stability_penalty(ARc, ar_lags)
          stability_penalty_ += stability_penalty(MAc, ma_lags)

      # Initialize residuals
      at = np.zeros((nT, K))
      for t in range(min(maxk, nT)):
          at[t, :] = data[t, :].copy()

      # Compute residuals with reasonable overflow protection
      overflow_threshold = 1e3  # Much more reasonable
      
      for t in range(1, min(maxk, nT)):
          tmp = data[t, :].copy()
          
          # AR part
          for j in range(1, min(t + 1, maxk + 1)):
              j_start = (j - 1) * K
              j_end = j * K
              if j_end <= ARc.shape[1]:
                  ar_contrib = ARc[:, j_start:j_end] @ data[t - j, :]
                  if np.any(np.abs(ar_contrib) > overflow_threshold):
                      numerical_penalty += penalty_scale * (np.max(np.abs(ar_contrib)) - overflow_threshold)**2
                  tmp -= ar_contrib
          
          # MA part  
          for j in range(1, min(t + 1, maxk + 1)):
              j_start = (j - 1) * K
              j_end = j * K
              if j_end <= MAc.shape[1]:
                  ma_contrib = MAc[:, j_start:j_end] @ at[t - j, :]
                  if np.any(np.abs(ma_contrib) > overflow_threshold):
                      numerical_penalty += penalty_scale * (np.max(np.abs(ma_contrib)) - overflow_threshold)**2
                  tmp -= ma_contrib
          
          at[t, :] = tmp

      # Build coefficient matrix
      beta_parts = []
      if ARc.shape[1] > 0:
          beta_parts.append(ARc.T)
      if MAc.shape[1] > 0:
          beta_parts.append(MAc.T)
      beta = np.vstack(beta_parts) if beta_parts else np.zeros((0, K))

      # Main loop
      for t in range(maxk, nT):
          Past = []
          for j in range(1, maxk + 1):
              if t - j >= 0:
                  Past.extend(data[t - j, :])
          for j in range(1, maxk + 1):
              if t - j >= 0:
                  Past.extend(at[t - j, :])
          
          Past = np.array(Past).reshape(1, -1)
          
          if beta.size > 0 and Past.shape[1] >= beta.shape[0]:
              y_hat = Past[:, :beta.shape[0]] @ beta
          else:
              y_hat = np.zeros(K)
          
          residual = data[t, :] - y_hat.flatten()
          at[t, :] = residual

      # Use residuals for likelihood
      at_use = at[maxk:, :]
      
      # Handle numerical issues
      if np.any(np.isnan(at_use)) or np.any(np.isinf(at_use)):
          return 1e10
      
      if np.any(np.abs(at_use) > 1e6): 
          return 1e10

      # Compute likelihood
      n_obs = nT - maxk
      if n_obs <= K:
          return 1e10
      
      sig = (at_use.T @ at_use) / n_obs
      # print(sig)
      # Light regularization
      reg_term = max(1e-10, np.mean(np.diag(sig)) * 1e-8)
      sig += reg_term * np.eye(K)
      
      try:
          eigenvals = np.linalg.eigvals(sig)
          if np.any(eigenvals <= 1e-12):
              return 1e10
          # Compute ACTUAL log-likelihood
          logdet = np.linalg.slogdet(sig)[1]
          if not np.isfinite(logdet):
              return 1e10
          # log L = -0.5 * n_obs * K * log(2π) - 0.5 * n_obs * log|Σ| - 0.5 * tr(Σ^(-1) * S)
          log_likelihood = -0.5 * n_obs * K * np.log(2 * np.pi)
          log_likelihood -= 0.5 * n_obs * logdet
          
          # Trace term
          sig_inv = inv(sig)
          S = at_use.T @ at_use  # Sum of squares matrix
          trace_term = np.trace(sig_inv @ S)
          log_likelihood -= 0.5 * trace_term
          
          if not np.isfinite(log_likelihood):
              return 1e10
              
      except (np.linalg.LinAlgError, ValueError, OverflowError):
          return 1e10
      
      # Debug output
      total_penalty = boundary_penalty + stability_penalty_ + numerical_penalty
      
      # print(f"Raw log-likelihood: {log_likelihood:.6f}")
      # print(f"Boundary penalty: {boundary_penalty:.6f}")
      # print(f"Stability penalty: {stability_penalty:.6f}")  
      # print(f"Numerical penalty: {numerical_penalty:.6f}")
      # print(f"Total penalty: {total_penalty:.6f}")
      
      # Return NEGATIVE log-likelihood plus penalties for minimization
      objective = -log_likelihood + total_penalty
      #print(f"Final objective (for minimizer): {objective:.6f}")
      
      return objective
      ############################################################
  ##Pure Loglikelihood and Hessiand comp:
  def pure_log_likelihood(self,par,enforce_sta_inver=False,p=None,q=None):
        """
        Pure log-likelihood WITHOUT penalties - for Hessian computation only
        This gives us the true curvature of the likelihood surface
        """
        T, K = self.data.shape
        data = np.array(self.data)
        maxk = np.max(self.Kronind) if self.structural_id else max(p, q)
        maxk = min(maxk, T - 1)
        nT = T
        par = np.array(par)
        
        # Parameter extraction (same logic as main function)
        if self.structural_id:
            Kpar = par.copy()
            Ph0 = np.eye(K)
            kp1 = self.AR_s.shape[1]  # Use self.AR_s instead of PhiID
            kp = kp1 - K
            PH = np.zeros((K, kp))
            Kidx = self.MA_s.shape[0]  # Use self.MA_s instead of ThetaID
            TH = np.zeros((Kidx, kp))
            ARid = self.AR_s
            MAid = self.MA_s
            icnt = 0

            for i in range(K):
                all_ar_idx = np.where(ARid[i, :] > 1)[0]
                all_ma_idx = np.where(MAid[i, :] > 1)[0]
                kdx = all_ar_idx[all_ar_idx < K]
                idx = all_ar_idx[all_ar_idx >= K]
                jdx = all_ma_idx[all_ma_idx >= K]

                if len(kdx) > 0:
                    Ph0[i, kdx] = Kpar[icnt:icnt + len(kdx)]
                    icnt += len(kdx)

                if len(idx) > 0:
                    col_positions = idx - K
                    valid_cols = (col_positions >= 0) & (col_positions < kp)
                    valid_idx = idx[valid_cols]
                    valid_positions = col_positions[valid_cols]
                    PH[i, valid_positions] = Kpar[icnt:icnt + len(valid_idx)]
                    icnt += len(valid_idx)

                if len(jdx) > 0:
                    col_positions = jdx - K
                    valid_cols = (col_positions >= 0) & (col_positions < kp)
                    valid_jdx = jdx[valid_cols]
                    valid_positions = col_positions[valid_cols]
                    TH[i, valid_positions] = Kpar[icnt:icnt + len(valid_jdx)]
                    icnt += len(valid_jdx)

            Ph0 += 1e-12 * np.eye(K)
            
            try:
                Ph0i = inv(Ph0)
            except (np.linalg.LinAlgError, ValueError):
                return -1e10

            ARc = Ph0i @ PH
            MAc = Ph0i @ TH

        else:
            kp = K**2 * p if p is not None else 0
            kq = K**2 * q if q is not None else 0
            ARc = par[:kp].reshape(K, p*K, order='F') if p > 0 else np.zeros((K, 0))
            MAc = par[kp:kp + kq].reshape(K, q * K, order='F') if q > 0 else np.zeros((K, 0))

        # Quick stability check - return bad value if unstable but don't add penalties
        if enforce_sta_inver:
            try:
                # Check AR stability
                if ARc.shape[1] > 0:
                    ar_lags = ARc.shape[1] // K
                    if ar_lags > 0:
                        companion = np.zeros((K * ar_lags, K * ar_lags))
                        if ar_lags > 1:
                            companion[K:, :-K] = np.eye(K * (ar_lags - 1))
                        for j in range(ar_lags):
                            companion[:K, j * K:(j + 1) * K] = ARc[:, j * K:(j + 1) * K]
                        
                        ar_eigvals = np.abs(np.linalg.eigvals(companion))
                        if np.any(ar_eigvals >= 0.999):
                            return -1e10
                
                # Check MA stability
                if MAc.shape[1] > 0:
                    ma_lags = MAc.shape[1] // K
                    if ma_lags > 0:
                        companion = np.zeros((K * ma_lags, K * ma_lags))
                        if ma_lags > 1:
                            companion[K:, :-K] = np.eye(K * (ma_lags - 1))
                        for j in range(ma_lags):
                            companion[:K, j * K:(j + 1) * K] = MAc[:, j * K:(j + 1) * K]
                        
                        ma_eigvals = np.abs(np.linalg.eigvals(companion))
                        if np.any(ma_eigvals >= 0.999):
                            return -1e10
            except:
                return -1e10

        # Compute residuals (same logic as main function)
        at = np.zeros((nT, K))
        for t in range(min(maxk, nT)):
            at[t, :] = data[t, :].copy()

        for t in range(1, min(maxk, nT)):
            tmp = data[t, :].copy()
            
            for j in range(1, min(t + 1, maxk + 1)):
                j_start = (j - 1) * K
                j_end = j * K
                if j_end <= ARc.shape[1]:
                    tmp -= ARc[:, j_start:j_end] @ data[t - j, :]
                if j_end <= MAc.shape[1]:
                    tmp -= MAc[:, j_start:j_end] @ at[t - j, :]
            
            at[t, :] = tmp

        # Build coefficient matrix
        beta_parts = []
        if ARc.shape[1] > 0:
            beta_parts.append(ARc.T)
        if MAc.shape[1] > 0:
            beta_parts.append(MAc.T)
        beta = np.vstack(beta_parts) if beta_parts else np.zeros((0, K))

        # Main loop
        for t in range(maxk, nT):
            Past = []
            for j in range(1, maxk + 1):
                if t - j >= 0:
                    Past.extend(data[t - j, :])
            for j in range(1, maxk + 1):
                if t - j >= 0:
                    Past.extend(at[t - j, :])
            
            Past = np.array(Past).reshape(1, -1)
            
            if beta.size > 0 and Past.shape[1] >= beta.shape[0]:
                y_hat = Past[:, :beta.shape[0]] @ beta
            else:
                y_hat = np.zeros(K)
            
            residual = data[t, :] - y_hat.flatten()
            at[t, :] = residual

        # Compute pure log-likelihood
        at_use = at[maxk:, :]
        
        if np.any(np.isnan(at_use)) or np.any(np.isinf(at_use)):
            return -1e10
        
        n_obs = nT - maxk
        if n_obs <= K:
            return -1e10
        
        sig = (at_use.T @ at_use) / n_obs
        # print(sig)
        reg_term = max(1e-12, np.mean(np.diag(sig)) * 1e-10)
        sig += reg_term * np.eye(K)
        
        try:
            eigenvals = np.linalg.eigvals(sig)
            if np.any(eigenvals <= 1e-12):
                return -1e10
                
            logdet = np.linalg.slogdet(sig)[1]
            if not np.isfinite(logdet):
                return -1e10
                
            # Pure log-likelihood computation
            log_likelihood = -0.5 * n_obs * K * np.log(2 * np.pi)
            log_likelihood -= 0.5 * n_obs * logdet
            
            sig_inv = inv(sig)
            S = at_use.T @ at_use
            trace_term = np.trace(sig_inv @ S)
            log_likelihood -= 0.5 * trace_term
            
            if not np.isfinite(log_likelihood):
                return -1e10
                
            return log_likelihood  # Return POSITIVE log-likelihood for Hessian

        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return -1e10
  def _estimate(self,par, LB, UB, structural_id=False, p=None, q=None,enforce_sta_inver=False,output=True):
      self.structural_id = structural_id
      if structural_id:
        logger.info(f"Structural mode: Setting p = q ")
      else:
        if p is None or q is None:
          raise ValueError("p and q must be specified in non-structural mode")
        if not (isinstance(p, int) and isinstance(q, int) and p >= 0 and q >= 0):
          raise ValueError(f"p and q must be non-negative integers, got p={p}, q={q}") 
      estimates,vals=minimize_qn(par,lambda par:self.log_likelihood(par=par,LB=LB,UB=UB,structural_id=structural_id,PhiID=self.AR_s, ThetaID=self.MA_s, Kron_index=self.Kronind,p=p,q=q,enforce_sta_inver=enforce_sta_inver),verbose=output)
      Log_lik=vals[3]    
      ##Hessian computation
      nump=len(estimates)
      eps=1e-3*estimates
      # np.where(estimates!= 0,estimates,1.0)
      Hessian = np.zeros((nump,nump))
      for i in range(nump):
        for j in range(nump):
          x1 = x2 = x3 = x4 = estimates.copy()
          x1[i]+=eps[i]
          x1[j]+=eps[j]
          x2[i]+=eps[i]  # Fixed: was eps[j]
          x2[j]+=eps[j]
          x3[i]-=eps[i]
          x3[j]+=eps[j]
          x4[i]-=eps[i]  # Fixed: was eps[j]
          x4[j]-=eps[j]
          ll1=self.pure_log_likelihood(x1,enforce_sta_inver,p,q)
          ll2=self.pure_log_likelihood(x2,enforce_sta_inver,p,q)
          ll3=self.pure_log_likelihood(x3,enforce_sta_inver,p,q)
          ll4=self.pure_log_likelihood(x4,enforce_sta_inver,p,q)
          Hessian[i,j]=(ll1-ll2-ll3+ll4)/(4*eps[i]*eps[j])
      de_t=det(Hessian)
      tol=1e-10
      if de_t<tol:
        logger.warning("Hessian is near-singular; standard errors set to ones")
        se_coeffs=np.ones(nump)
      else:
        se_coeffs=np.sqrt(np.diag(inv(Hessian)))
      t_val=estimates/se_coeffs
      p_val=2*(1-norm.cdf(np.abs(t_val)))
      # print(len(self.data.columns))
      k=len(self.data.columns)
      max_qp = np.max(self.Kronind) if structural_id else max(p, q)
      kp=k*max_qp
      AR_l0=np.eye(k) if structural_id else np.zeros((k,k))
      se_AR_l0=np.zeros_like(AR_l0)
      AR=np.zeros((k,kp))
      se_AR=np.zeros((k,kp))
      MA = np.zeros((k, kp))
      se_MA = np.zeros((k, kp))

      coun=0
      if structural_id:
        for i in range(k):
          idx = np.where(self.AR_s[i, :] > 1)[0]
          jdx = np.where(self.MA_s[i, :] > 1)[0]
          kdx = np.where(self.AR_s[i, :k] > 1)[0]
          if kdx.size>0:
            idx=np.setdiff1d(idx,kdx)
            jdx=np.setdiff1d(jdx,kdx)  # Fixed: was setfigg1d
            AR_l0[i,kdx]=estimates[coun:coun+len(kdx)]
            se_AR_l0[i,kdx]=se_coeffs[coun:coun+len(kdx)]
            coun+=len(kdx)
          if idx.size>0:
            col_positions=idx-k
            valid_cols=(col_positions>=0) & (col_positions < kp)
            valid_idx=idx[valid_cols]
            valid_positions=col_positions[valid_cols]
            for col_pos,param_idx in zip(valid_positions,range(coun,coun+len(valid_idx))):
              AR[i,col_pos]=estimates[param_idx]
              se_AR[i,col_pos]=se_coeffs[param_idx]
            coun+=len(valid_idx)
          if jdx.size>0:
            col_positions=jdx-k
            valid_cols=(col_positions >= 0) & (col_positions < kp)
            valid_jdx = jdx[valid_cols]
            valid_positions = col_positions[valid_cols]
            for col_pos, param_idx in zip(valid_positions, range(coun,coun+len(valid_jdx))):
              MA[i, col_pos] = estimates[param_idx]
              se_MA[i, col_pos] = se_coeffs[param_idx]
            coun+=len(valid_jdx)
      else:
        # Non-structural mode: parameters are organized as [AR_params, MA_params]
        # AR_params: K^2*p parameters, MA_params: K^2*q parameters
        
        if p > 0:
          # AR parameters: reshape from flat array to (K, p*K) matrix
          ar_params = estimates[coun:coun + k*k*p]
          AR[:, :p*k] = ar_params.reshape(k, p*k, order='F')
          se_ar_params = se_coeffs[coun:coun + k*k*p]
          se_AR[:, :p*k] = se_ar_params.reshape(k, p*k, order='F')
          coun += k*k*p
        
        if q > 0:
          # MA parameters: reshape from flat array to (K, q*K) matrix  
          ma_params = estimates[coun:coun + k*k*q]
          MA[:, :q*k] = ma_params.reshape(k, q*k, order='F')
          se_ma_params = se_coeffs[coun:coun + k*k*q]
          se_MA[:, :q*k] = se_ma_params.reshape(k, q*k, order='F')
          coun += k*k*q

      ####Residual
      T,K=self.data.shape
      data=self.data.to_numpy()
      if T<=max_qp:
        raise ValueError(f"Data length ({T}) must be greater than max_qp ({max_qp})")
      try:
        AR_0_i=inv(AR_l0) if structural_id else np.eye(k)
      except np.linalg.LinAlgError:
        logger.warning("AR_l0 matrix is singular; using pseudo_inv")
        AR_0_i=np.linalg.pinv(AR_l0)

      AR_c=AR_0_i@AR
      MA_c=AR_0_i@MA
      res=np.zeros((T,k))
      maxx=max_qp

      if max_qp>0:
        res[0,:]=data[0,:]
        for t in range(1,max_qp):
          tmp=data[t,:].copy()  # Fixed: added parentheses
          for j in range(1,t+1):
            if t-j >=0 and (j-1)*k < AR_c.shape[1]:
              j_start=(j-1)*k
              j_end=j_start+k
              tmp-=AR_c[:,j_start:j_end]@res[t-j,:]
            if t-j>0 and (j-1)*k < MA_c.shape[1]:
              j_start=(j-1)*k
              j_end=j_start+k
              tmp-=MA_c[:,j_start:j_end]@res[t-j,:]
          res[t,:]=tmp
      for t in range(max_qp,T):
        P_ar = []
        P_ma = []
        for j in range(1, max_qp+1):
          if t-j >= 0:
            P_ar.extend(data[t-j,:])
          else:
            P_ar.extend(np.zeros(k))
        
        # Collect MA terms (lagged residuals)
        for j in range(1, max_qp+1):
          if t-j >= 0:
            P_ma.extend(res[t-j,:])
          else:
            P_ma.extend(np.zeros(k))
        
        # Combine AR and MA terms
        P = np.array(P_ar + P_ma).reshape(1, -1)
        
        # Construct beta matrix to match P dimensions
        beta = np.vstack([AR_c.T, MA_c.T])
        
        # Ensure dimensions match
        if P.shape[1] != beta.shape[0]:
          # Adjust beta or P to match dimensions
          min_dim = min(P.shape[1], beta.shape[0])
          P = P[:, :min_dim]
          beta = beta[:min_dim, :]
        
        tmp = P @ beta
        res[t,:] = data[t,:] - tmp.flatten()

      ###Numerical issues check:
      if np.any(np.isnan(res)) or np.any(np.isinf(res)):
        logger.warning("NaN or Inf values detected in residuals; results may be unreliable")

      residual=res[max_qp:T,:]  
      fitted_d=data[max_qp:T,:]

      ###AIC,BIC
      sigs=(residual.T@residual)/(T-max_qp)
      # print("sigs",sigs)
      if np.any(np.isnan(sigs)) or np.any(np.isinf(sigs)):
        logger.warning("Invalid residual covariance matrix; adding perturbation")
        sigs += 1e-6* np.eye(k)

      try:
        d=np.linalg.det(sigs)
        d_=np.log(d) if d >1e-6 else np.log(1e-6)
      except np.linalg.LinAlgError:
        logger.warning("Failed to compute determinant of residual covariance; using default")
        d_ = np.log(tol)
      # print(d_)      
      aic=d_+2*nump/T
      bic=d_+np.log(T)*nump/T
      logger.info(f"AIC: {aic:.4f}, BIC: {bic:.4f}")
      return{'log_lik':Log_lik,'aic':aic,'bic': bic,'pvalue': p_val,'tvalue': t_val,
                'estimates': estimates,
                'se': se_coeffs,
                'residuals': residual,
                'fitted': fitted_d,
                'MA/AR_0':AR_l0,
                'AR': AR,
                'MA': MA,
                'se_L0': se_AR_l0,
                'se_AR': se_AR,
                'se_MA': se_MA,
                'p': p,
                'q':q}
  def display_results(self,results):
    strr='with structural identification:'if self.structural_id else ''
    print("=" * 90)
    if self.best_p==0:
      stri=f"Estimates for VMA{strr}: VMA({self.best_q})"
    elif self.best_q==0 :
      stri=f"Estimates for VAR{strr}: VAR({self.best_p})"
    else:
      stri=f"Estimates for VARMA{strr}: VARMA({self.best_p},{self.best_q})"
    print(stri)
    print("=" * 90)

    # Model fit statistics
    print(f"Log-likelihood: {results['log_lik']:.6f}")
    print(f"AIC: {results['aic']:.6f}")
    print(f"BIC: {results['bic']:.6f}")
    print("=" * 90)

    # Get dimensions
    K = self.data.shape[1]

    # Parameter table header
    print(f"{'Parameter':<25} | {'Value':>12} | {'Std.Err':>12} | {'T-value':>10} | {'P-value':>10}")
    print("-" * 90)

    # Display AR/MA lag 0 (contemporaneous parameters)
    if 'MA/AR_0' in results and results['MA/AR_0'] is not None and not np.all(results['MA/AR_0'] == 0):
        print("AR/MA Lag 0:")
        AR_l0 = results['MA/AR_0']
        se_l0 = results['se_L0']
        #print(AR_l0)
        param_idx = 0
        for i in range(K):
            for j in range(K):
                val = AR_l0[i, j]
                se_val = se_l0[i, j]
                param_name = f"  L0[{i+1},{j+1}]"
                t_val = float(self.best_model['tvalue'][param_idx]) if param_idx < len(self.best_model['tvalue']) else 0.0
                p_val = float(self.best_model['pvalue'][param_idx]) if param_idx < len(self.best_model['pvalue']) else 1.0
                param_idx += 1
                
                print(f"{param_name:<25} | {val:>12.6f} | {se_val:>12.6f} | {t_val:>10.4f} | {p_val:>10.4f}")
        print()

    # Display AR parameters
    if 'AR' in results and results['AR'] is not None and results['AR'].size > 0:
        print("Autoregressive Parameters:")
        AR = results['AR']
        se_AR = results['se_AR']
        
        # Calculate starting parameter index (after L0 parameters)
        param_idx = K * K  # Number of L0 parameters
        
        # AR is K x (K*max_p) matrix, where max_p is number of AR lags
        num_ar_lags = AR.shape[1] // K
        
        for lag in range(1, num_ar_lags + 1):
            print(f"AR Lag {lag}:")
            
            for i in range(K):
                for j in range(K):
                    col_idx = (lag-1) * K + j
                    if col_idx < AR.shape[1]:
                        val = AR[i, col_idx]
                        se_val = se_AR[i, col_idx]
                        param_name = f"  AR{lag}[{i+1},{j+1}]"
                        t_val = float(self.best_model['tvalue'][param_idx]) if param_idx < len(self.best_model['tvalue']) else 0.0
                        p_val = float(self.best_model['pvalue'][param_idx]) if param_idx < len(self.best_model['pvalue']) else 1.0
                        param_idx += 1
                        
                        print(f"{param_name:<25} | {val:>12.6f} | {se_val:>12.6f} | {t_val:>10.4f} | {p_val:>10.4f}")
            print()

    # Display MA parameters
    if 'MA' in results and results['MA'] is not None and len(results['MA']) > 0:
        print("Moving Average Parameters:")
        MA = results['MA']
        se_MA = results['se_MA']
        
        params_per_lag = K * K
        num_ma_lags = params_per_lag//len(MA) 
        
        # Calculate starting parameter index (after L0 and AR parameters)
        param_idx = K * K + len(results.get('AR', []))
        
        for lag in range(1, num_ar_lags + 1):
          print(f"MA Lag {lag}:")
          for i in range(K):
            for j in range(K):
              col_idx = (lag-1) * K + j
              if col_idx < MA.shape[1]:
                  val = MA[i, col_idx]
                  se_val = se_MA[i, col_idx]
                  param_name = f"  MA{lag}[{i+1},{j+1}]"
                  t_val = float(self.best_model['tvalue'][param_idx]) if param_idx < len(self.best_model['tvalue']) else 0.0
                  p_val = float(self.best_model['pvalue'][param_idx]) if param_idx < len(self.best_model['pvalue']) else 1.0
                  param_idx += 1            
                  print(f"{param_name:<25} | {val:>12.6f} | {se_val:>12.6f} | {t_val:>10.4f} | {p_val:>10.4f}")
        print()
    print("=" * 90)
  def fit(self,p=None,q=None,output=True):
    K=len(self.columns)
    if p is not None and (not isinstance(p, int) or p < 0):
      raise ValueError("p must be a non-negative integer.")
    if q is not None and (not isinstance(q, int) or q < 0):
      raise ValueError("q must be a non-negative integer.")
    #suppose user init the VARMA with s_id=T and self.mode='general'=> 'no struct id':
    if self.structural_id and self.key=='general':
      user_input=input("Do you want to proceed with Structural Id? (Y/N): ").strip().upper()
      if user_input == 'Y':
        self.structural_id = True
        self.mode = ''
      elif user_input != 'N':
        print("Defaulting to non-structural identification.")
        self.structural_id = False 
    ###initialisation:
    if self.structural_id:
      estims,s_e=self._ini_s1(output=output)
      par, separ, lB, uB=self._prepare_for_est(estims,s_e,output)
      results= self._estimate(par=par,LB=lB,UB=uB,p=p, q=q, structural_id=True,output=output)
      self.best_model = {
          'p': self.best_p, 
          'q': self.best_q, 
          'aic': results['aic'], 
          'bic': results['bic'],
          'MA/AR_l0': results['MA/AR_0'],
          "AR": results['AR'], 
          "MA": results['MA'], 
          "se_L0": results['se_L0'], 
          "se_AR": results['se_AR'], 
          "se_MA": results['se_MA'],
          'fitted': results['fitted'], 
          'residuals': results['residuals'], 
          'beta': results['estimates'],
          'se': results['se'], 
          'tvalue': results['tvalue'], 
          'pvalue': results['pvalue'],
          'log_lik':results['log_lik']
      }
    else:
      res=[]
      def evls(p,q):
        try:
          estims,s_e=self._ini_s1(p=p,q=q,output=output)
          #print(estims)
          par, separ, lB, uB=self._prepare_for_est(estims,s_e,output)
          resu=self._estimate(par=par,LB=lB,UB=uB,p=p, q=q, structural_id=False,output=output)
          #print(resu)
          return resu
        except Exception as e:
          logger.warning(f"Failed for p={p}, q={q}: {e}")
          print(f"Failed for p={p}, q={q}: {e}")
          return None
      # Try special (0,1) and (1,0) cases first
      res = []
      for p, q in [(0, 1), (1, 0)]:
          r = evls(p, q)
          if r:
              res.append(r)
      # Now parallelize the rest
      def safe_evls(p, q):
          try:
              r = evls(p, q)
              return r
          except Exception as e:
              print(f"Failed for p={p}, q={q}: {e}")
              return None
      # result = Parallel(n_jobs=-1)(
      #     delayed(safe_evls)(p, q)
      #     for p in range(1, self.max_p + 1)
      #     for q in range(1, self.max_q + 1)
      #     if (p, q) not in [(0, 1), (1, 0)]
      # )

        
      for p in range(1,self.max_p + 1):
        for q in range(1,self.max_q + 1):
          print(p,q)
          result = evls(p, q)
          res.append(result)
      # ersults = [r for r in res if r is not None]
      # print(ersults)
      #Display formatted results
      res= [r for r in res if r is not None]
      # res += result
      # Pick the best model
      if res:
          criterion_key = 'aic' if self.criterion.lower() == 'aic' else 'bic'
          results= min(res, key=lambda x: x[criterion_key])
          self.best_model = results
          self.best_p=results['p']
          self.best_q=results['q']
      else:
          self.best_model = None   
      # if self.max_p >1 and self.max_q >1 :
      #   result=Parallel(n_jobs=-1)(delayed(evls)(p,q) for p in range(1,self.max_p+1) for q in range(1,self.max_q+1))
      #   if result:
      #     criterion_key = 'aic' if self.criterion.lower() == 'aic' else 'bic'
      #     results=min(result,key=lambda x:x[criterion_key])  
      # else:
      #   if p is not None and q is not None:
      #     results=evls(p,q)
      
      # for p in range(1,self.max_p + 1):
      #   for q in range(1,self.max_q + 1):
      #     print(p,q)
      #     result = evls(p, q)
      #     res.append(result)
      # ersults = [r for r in res if r is not None]
      # print(ersults)
      #Display formatted results


    if self.best_model:
      #check if the model passes :
      if output:
        self.display_results(results)
      self.run_full_diagnosis(plot=output,threshold=self.Threshold)
      logger.info("Best Model: ")
      print(f"AIC: {self.best_model['aic']:.6f}")
      print(f"BIC: {self.best_model['bic']:.6f}")
      print(f'VARMA({self.best_p},{self.best_q})')
      if output:

        logger.info("Generating fitted vs actual plots")
        fitted = self.best_model['fitted']
        train_data = self.data.iloc[self.best_model['p']:]
        if fitted.shape[0] != len(train_data):
            raise ValueError(f"Fitted values shape {fitted.shape} does not match training data length {len(train_data)}")
        
        fitted_df = pd.DataFrame(fitted, index=train_data.index, columns=self.data.columns)
        n_vars = K
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
        axes = np.array(axes).flatten() if n_vars > 1 else [axes]
        
        for i, col in enumerate(self.data.columns):
            ax = axes[i]
            ax.plot(train_data.index, train_data[col], 'b-', label='Original Train Data', linewidth=1.5)
            ax.plot(fitted_df.index, fitted_df[col], 'r--', label='VARMA Fitted Values', linewidth=1.5)
            ax.set_title(f'{col}: Original vs Fitted')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for j in range(n_vars, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()
        model=self.best_model
        return model
      else:
        return self.best_model
    else:
      raise ValueError("No Model was fitted")
  def run_full_diagnosis(self, num_lags=8, plot=True, threshold=0.8):
      if not 0 <= threshold <= 1:
          raise ValueError("Threshold needs to be between 0 and 1")
      Diagnosis = {}
      K = len(self.data.columns)
      if not hasattr(self, 'best_model') or self.best_model is None:
          print("No model fitted. Cannot perform diagnostics.")
          Diagnosis['Final Diagnosis'] = 'Not Fitted'
          return Diagnosis
      resids = self.best_model['residuals']
      if resids.shape[1] != K:
          raise ValueError(f"Residuals have {resids.shape[1]} columns, expected {K}")
      if resids.shape[0] < num_lags:
          print(f"Warning: Sample size ({resids.shape[0]}) < num_lags ({num_lags})")
          num_lags=resids.shape[0]/3
      print("===================Stability===========================")
      cm_results = self.companion_matrix(plot=False, output=False)
      cm = cm_results['companion_matrix']
      is_stable = cm_results['is_stable']
      eigs = cm_results['eigenvalues']
      max_eigenvalue = cm_results['max_eigenvalue']
      print(f"The VARMA model is stable: {max_eigenvalue:.4f} ({'Stable' if is_stable else 'Not Stable'})")
      Diagnosis['Final Stability Diagnosis'] = 'Stable' if is_stable else "Not Stable"
      S_score = 2 if is_stable else 0
      Diagnosis['Stability Score'] = S_score
      print("===================Invertibility===========================")
      p = self.best_p
      q = self.best_q
      is_invertible = True
      max_ma_eigenvalue = np.nan
      ma_eigs = np.array([])
      if q > 0:
          MA = self.best_model.get('MA', np.zeros((K, K*q)))
          if self.structural_id:
              AR_l0 = self.best_model.get('MA/AR_l0', np.eye(K))
              try:
                  AR_l0_inv = inv(AR_l0)
              except np.linalg.LinAlgError:
                  AR_l0_inv = np.linalg.pinv(AR_l0)
                  logger.warning("AR_l0 matrix is singular, using pseudoinverse")
              MA_reduced = AR_l0_inv @ MA
          else:
              MA_reduced = MA
          ma_companion_dim = K * q
          MA_companion = np.zeros((ma_companion_dim, ma_companion_dim))
          for i in range(q):
              start_col = i * K
              end_col = (i + 1) * K
              if end_col <= MA_reduced.shape[1]:
                  MA_companion[:K, start_col:end_col] = MA_reduced[:, start_col:end_col]
          if q > 1:
              MA_companion[K:, :K*(q-1)] = np.eye(K*(q-1))
          try:
              ma_eigs = eigvals(MA_companion)
              max_ma_eigenvalue = np.max(np.abs(ma_eigs))
              is_invertible = max_ma_eigenvalue < 1.0
          except np.linalg.LinAlgError:
              logger.warning("Could not compute MA eigenvalues")
              is_invertible = False
              max_ma_eigenvalue = np.nan
              ma_eigs = np.array([])
      print(f"The VARMA model is invertible: {max_ma_eigenvalue:.4f} ({'Invertible' if is_invertible else 'Not Invertible'})")
      Diagnosis['Final Invertibility Diagnosis'] = 'Invertible' if is_invertible else "Not Invertible"
      I_score = 2 if is_invertible else 0
      Diagnosis['Invertibility Score'] = I_score
      print("===================Serial Correlation Tests===========================")
      lb_results = []
      dw_results = []
      print(f"===============Ljung–Box Test (lags={num_lags})==============")
      for i, col in enumerate(self.data.columns):
          lb_test = acorr_ljungbox(resids[:, i], lags=[num_lags], return_df=True)
          pval = lb_test['lb_pvalue'].values[0]
          LbT = "PASS" if pval > 0.05 else "FAIL"
          print(f"Residual {col}: p-value = {pval:.4f} → {LbT}")
          lb_results.append(LbT)
      print("==================Durbin-Watson Statistics=================")
      for i, col in enumerate(self.data.columns):
          dw = durbin_watson(resids[:, i])
          dw_result = "Pass" if 1.5 <= dw <= 2.5 else "Fail"
          print(f"Residual {col}: DW = {dw:.4f} → {dw_result}")
          dw_results.append(dw_result)
      DW_score = dw_results.count('Pass') / K
      LB_score = lb_results.count('PASS') / K
      auto_corr_score = 0
      for dw_res, lb_res in zip(dw_results, lb_results):
          tests_passed = 0
          if dw_res == "Pass":
              tests_passed += 1
          if lb_res == "PASS":
              tests_passed += 1
          auto_corr_score += tests_passed / 2
      auto_corr_score /= K
      Diagnosis['DW_score'] = DW_score
      Diagnosis['LB_score'] = LB_score
      Diagnosis['Autocorrelation_score'] = auto_corr_score
      Diagnosis['DW_diagnosis'] = 'Passed' if DW_score == 1 else 'Failed'
      Diagnosis['LB_diagnosis'] = 'Passed' if LB_score == 1 else 'Failed'
      Diagnosis['Autocorrelation_diagnosis'] = 'Passed' if auto_corr_score == 1 else 'Failed'
      Diagnosis['Final Autocorrelation Diagnosis'] = 'Passed' if DW_score == 1 and LB_score == 1 else 'Failed'
      print("==================Heteroskedasticity=================")
      Homoscedasticity = True
      arch_res = []
      for i, col in enumerate(self.data.columns):
          arch_test = het_arch(resids[:, i])
          arch_res.append('pass' if arch_test[1] >= 0.05 else 'Fail')
          print(f"Residual {col}: ARCH p-value = {arch_test[1]:.4f} → {arch_res[i]}")
      arch_tests = arch_res.count('pass') / K
      if arch_tests != 1:
          Homoscedasticity = False
      Diagnosis['Heteroskedasticity_score'] = arch_tests
      Diagnosis['Heteroskedasticity_diagnosis'] = 'Passed' if arch_tests == 1 else 'Failed'
      Diagnosis['Final Heteroskedasticity Diagnosis'] = 'Passed' if Homoscedasticity else 'Failed'
      print("=======================Normality Test=======================")
      Normality = True
      jb_results = []
      shapiro_results = []
      for i, col in enumerate(self.data.columns):
          jb_test = jarque_bera(resids[:, i])
          sh_test = shapiro(resids[:, i])
          jb_pval = jb_test[1]
          sh_pval = sh_test[1]
          print(f"Residual {col}: JB p-value = {jb_pval:.4f}, Shapiro p-value = {sh_pval:.4f}")
          jb_results.append('pass' if jb_pval >= 0.05 else 'fail')
          shapiro_results.append('pass' if sh_pval >= 0.05 else 'fail')
      joint_passes = sum(1 for j, s in zip(jb_results, shapiro_results) if j == 'pass' and s == 'pass')
      normality_score = joint_passes / K
      if normality_score != 1:
          Normality = False
      Diagnosis['Normality_score'] = normality_score
      Diagnosis['Normality_diagnosis'] = 'Passed' if normality_score == 1 else 'Failed'
      Diagnosis['Final Normality Diagnosis'] = 'Passed' if Normality else 'Failed'
      print("=======================Structural Breaks========================")
      No_Structural_breaks = True
      cusum_stat, cusum_pval, _ = breaks_cusumolsresid(resids)
      print(f"CUSUM p-value: {cusum_pval:.4f}")
      if cusum_pval < 0.05:
          No_Structural_breaks = False
      print("No structural breaks detected" if No_Structural_breaks else "Structural breaks detected")
      Diagnosis['Final Structural Breaks'] = 'Passed' if No_Structural_breaks else 'Failed'
      structural_breaks_score = 1.0 if No_Structural_breaks else 0.0
      final_score = (Diagnosis['DW_score'] + Diagnosis['LB_score'] +
                    Diagnosis['Autocorrelation_score'] +
                    Diagnosis['Heteroskedasticity_score'] +
                    Diagnosis['Normality_score'] + structural_breaks_score +
                    S_score + I_score) / 8
      self.fitted = final_score >= threshold
      Diagnosis['Final_score'] = final_score
      Diagnosis['Verdict'] = 'Passed' if self.fitted else 'Failed'
      print("\n==================Diagnostic Summary=================")
      summary_table = {
          'Estimation': 'Maximum log-Likelihood' if self.structural_id else 'OLS',
          'Model': f'VARMA({self.best_model["p"]},{self.best_model["q"]})',
          'Log-Likelihood': self.best_model.get('log_lik', 'N/A'),
          'AIC': self.best_model.get('aic', 'N/A'),
          'BIC': self.best_model.get('bic', 'N/A'),
          'Stability': f"{Diagnosis['Stability Score']:.4f} ({Diagnosis['Final Stability Diagnosis']})",
          'Invertibility': f"{Diagnosis['Invertibility Score']:.4f} ({Diagnosis['Final Invertibility Diagnosis']})",
          'DW Score': f"{Diagnosis['DW_score']:.4f} ({Diagnosis['DW_diagnosis']})",
          'LB Score': f"{Diagnosis['LB_score']:.4f} ({Diagnosis['LB_diagnosis']})",
          'Autocorrelation Score': f"{Diagnosis['Autocorrelation_score']:.4f} ({Diagnosis['Autocorrelation_diagnosis']})",
          'Heteroskedasticity Score': f"{Diagnosis['Heteroskedasticity_score']:.4f} ({Diagnosis['Heteroskedasticity_diagnosis']})",
          'Normality Score': f"{Diagnosis['Normality_score']:.4f} ({Diagnosis['Normality_diagnosis']})",
          'Structural Breaks': f"{structural_breaks_score:.4f} ({Diagnosis['Final Structural Breaks']})",
          'Final Score': f"{final_score:.4f}",
          'Verdict': Diagnosis['Verdict']
      }
      print("Model Diagnostics Summary:")
      print("-" * 50)
      for key, value in summary_table.items():
          print(f"{key:<30} | {value}")
      print("-" * 50)
      if plot:
          logger.info("Generating diagnostic plots")
          T, K = resids.shape
          fig_height = 4 * (K + 2)
          fig, axes = plt.subplots(nrows=K + 2, ncols=2, figsize=(12, fig_height))
          flat_resid = resids.flatten()
          centered = flat_resid - np.mean(flat_resid)
          cusum = np.cumsum(centered)
          c = 0.948
          cusum_threshold = c * np.sqrt(len(cusum))
          ax_cusum = plt.subplot2grid((K + 2, 2), (0, 0), colspan=2)
          ax_cusum.plot(cusum, label='Global CUSUM of Residuals')
          ax_cusum.axhline(y=cusum_threshold, color='red', linestyle='--', label='+95% Band')
          ax_cusum.axhline(y=-cusum_threshold, color='red', linestyle='--', label='-95% Band')
          ax_cusum.axhline(y=0, color='black', linestyle='-', label='Zero Line')
          ax_cusum.set_title("Global CUSUM Test (All Residuals)")
          ax_cusum.set_xlabel("Flattened Time Index")
          ax_cusum.set_ylabel("CUSUM Value")
          ax_cusum.legend()
          ax_eigs = plt.subplot2grid((K + 2, 2), (1, 0), colspan=2)
          ax_eigs.scatter(eigs.real, eigs.imag, color='blue', alpha=0.5, label='AR Eigenvalues')
          if q > 0:
              ax_eigs.scatter(ma_eigs.real, ma_eigs.imag, color='green', alpha=0.5, label='MA Eigenvalues')
          circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--')
          ax_eigs.add_artist(circle)
          ax_eigs.axhline(0, color='black', linestyle='--', alpha=0.3)
          ax_eigs.axvline(0, color='black', linestyle='--', alpha=0.3)
          ax_eigs.set_title(f"Eigenvalues\nAR Max |λ| = {max_eigenvalue:.3f} ({'Stable' if is_stable else 'Unstable'}), "
                          f"MA Max |λ| = {max_ma_eigenvalue:.3f} ({'Invertible' if is_invertible else 'Not Invertible'})")
          ax_eigs.set_xlabel("Real Part")
          ax_eigs.set_ylabel("Imaginary Part")
          ax_eigs.legend()
          ax_eigs.grid(True)
          for i, col in enumerate(self.data.columns):
              ax_hist = plt.subplot2grid((K + 2, 2), (i + 2, 0))
              ax_hist.hist(resids[:, i], bins=30, density=True, alpha=0.7, color='steelblue')
              ax_hist.set_title(f"Histogram of Residual {col}")
              ax_hist.set_xlabel("Residual Value")
              ax_hist.set_ylabel("Density")
              ax_qq = plt.subplot2grid((K + 2, 2), (i + 2, 1))
              probplot(resids[:, i], dist="norm", plot=ax_qq)
              ax_qq.set_title(f"Q-Q Plot for Residual {col}")
          plt.tight_layout()
          plt.show()
      return Diagnosis    
  def companion_matrix(self, plot=True, output=True):
      if self.best_model is None:
          raise ValueError("Model must be fitted first.")
      
      K = len(self.columns)
      p = self.best_p
      q = self.best_q
      max_lag = max(p, q, 1)
      
      # Extract AR coefficients
      if self.structural_id:
          AR_l0 = self.best_model.get('MA/AR_l0', np.eye(K))
          AR = self.best_model.get('AR', np.zeros((K, K * p)))
          try:
              AR_l0_inv = inv(AR_l0)
          except np.linalg.LinAlgError:
              logger.warning("AR_l0 matrix is singular, using pseudoinverse")
              AR_l0_inv = np.linalg.pinv(AR_l0)
          A = AR_l0_inv @ AR
      else:
          A = self.best_model.get('AR', np.zeros((K, K * p)))
      
      # Reshape AR coefficients
      Phi = [A[:, i * K:(i + 1) * K].reshape(K, K) for i in range(p)] if A.size > 0 else [np.zeros((K, K)) for _ in range(max_lag)]
      
      # Construct companion matrix
      companion_dim = K * max_lag
      cm = np.zeros((companion_dim, companion_dim))
      for i in range(min(p, max_lag)):
          cm[:K, i * K:(i + 1) * K] = Phi[i]
      if max_lag > 1:
          cm[K:, :K * (max_lag - 1)] = np.eye(K * (max_lag - 1))
      
      # Compute eigenvalues and stability
      try:
          eigvals_cm = eigvals(cm)
          max_eig = np.max(np.abs(eigvals_cm))
          is_stable = max_eig < 1.0 - 1e-6
      except np.linalg.LinAlgError:
          logger.warning("Failed to compute eigenvalues")
          eigvals_cm = np.array([])
          max_eig = np.nan
          is_stable = False
      
      results = {
          'companion_matrix': cm,
          'eigenvalues': eigvals_cm,
          'is_stable': is_stable,
          'max_eigenvalue': max_eig
      }
      
      if output:
          print("=" * 80)
          print(f"VARMA({p},{q}) Companion Matrix Analysis")
          print(f"Variables: {K}")
          print(f"Companion matrix dimension: {companion_dim} x {companion_dim}")
          print(f"Maximum eigenvalue (modulus): {max_eig:.6f}")
          print(f"Stability: {'STABLE' if is_stable else 'UNSTABLE'}")
          print("=" * 80)
      
      if plot:
          plt.figure(figsize=(8, 6))
          plt.scatter(eigvals_cm.real, eigvals_cm.imag, color='blue', alpha=0.5, label='Eigenvalues')
          circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--')
          plt.gca().add_artist(circle)
          plt.axhline(0, color='black', linestyle='--', alpha=0.3)
          plt.axvline(0, color='black', linestyle='--', alpha=0.3)
          plt.title(f"Companion Matrix Eigenvalues\nMax |λ| = {max_eig:.3f} ({'Stable' if is_stable else 'Unstable'})")
          plt.xlabel("Real Part")
          plt.ylabel("Imaginary Part")
          plt.legend()
          plt.grid(True)
          plt.tight_layout()
          plt.show()
      
      return results

  def compute_irf(self, horizon=10, orth=True, bootstrap=True, n_boot=1000, plot=True, tol=1e-6):
      if self.best_model is None:
          raise ValueError("No model fitted. Cannot compute IRF.")
      
      K = len(self.columns)
      p = self.best_p
      q = self.best_q
      max_lag = max(p, q, 1)
      
      # Extract AR and MA coefficients
      if self.structural_id:
          AR_l0 = self.best_model.get('MA/AR_l0', np.eye(K))
          AR = self.best_model.get('AR', np.zeros((K, K * p)))
          MA = self.best_model.get('MA', np.zeros((K, K * q)))
          try:
              AR_l0_inv = inv(AR_l0)
          except np.linalg.LinAlgError:
              logger.warning("AR_l0 matrix is singular, using pseudoinverse")
              AR_l0_inv = np.linalg.pinv(AR_l0)
          A = AR_l0_inv @ AR
          M = AR_l0_inv @ MA
      else:
          A = self.best_model.get('AR', np.zeros((K, K * p)))
          M = self.best_model.get('MA', np.zeros((K, K * q)))
      
      # Reshape coefficients
      Phi = [A[:, i * K:(i + 1) * K].reshape(K, K) for i in range(p)] if A.size > 0 else [np.zeros((K, K)) for _ in range(max_lag)]
      Theta = [M[:, j * K:(j + 1) * K].reshape(K, K) for j in range(q)] if M.size > 0 else [np.zeros((K, K)) for _ in range(max_lag)]
      
      # Compute MA representation coefficients (ψ_h)
      psi = [np.eye(K)]  # h=0
      for h in range(1, horizon + 1):
          temp = np.zeros((K, K))
          for i in range(p):
              if h - (i + 1) >= 0:
                  temp += Phi[i] @ psi[h - (i + 1)]
          if h <= q:
              temp += Theta[h - 1]
          psi.append(temp)
      
      Psi = np.array(psi)  # shape (horizon+1, K, K)
      
      # Orthogonalize IRFs if requested
      if orth:
          Sigma = np.cov(self.best_model['residuals'].T)
          try:
              P = cholesky(Sigma)
              irf = np.array([Psi[i] @ P for i in range(horizon + 1)])
          except np.linalg.LinAlgError:
              logger.warning("Cholesky decomposition failed; using non-orthogonal IRFs")
              irf = Psi
              orth = False
      else:
          irf = Psi
      
      # Bootstrap for confidence intervals
      ci_lower = None
      ci_upper = None
      if bootstrap:
          boot_irfs = np.zeros((n_boot, horizon + 1, K, K))
          residuals = self.best_model['residuals']
          T = residuals.shape[0]
          data = self.data.values
          
          for b in range(n_boot):
              try:
                  boot_idx = np.random.choice(T, size=T, replace=True)
                  boot_resids = residuals[boot_idx]
                  Y_sim = np.zeros((T + max_lag, K))
                  Y_sim[:max_lag] = data[-max_lag:]
                  
                  for t in range(max_lag, T + max_lag):
                      Y_t = np.zeros(K)
                      for i in range(min(p, max_lag)):
                          if t - i - 1 >= 0 and i < len(Phi):
                              Y_t += Phi[i] @ Y_sim[t - i - 1]
                      for j in range(min(q, max_lag)):
                          if t - j - 1 >= 0 and j < len(Theta):
                              Y_t += Theta[j] @ boot_resids[t - max_lag - j]
                      Y_sim[t] = Y_t
                  
                  Y_sim = Y_sim[max_lag:]
                  X, Y = self.lag_matrix(max_lag)
                  boot_beta, _, _, _ = ols_estimator(X, Y_sim, tol=tol)
                  
                  # Reconstruct AR coefficients for bootstrap
                  boot_A = boot_beta[1:] if boot_beta.shape[0] == K * max_lag + 1 else boot_beta
                  boot_A = boot_A.reshape(max_lag, K, K).transpose(1, 2, 0) if boot_A.size > 0 else np.zeros((K, K, max_lag))
                  boot_Phi = [boot_A[:, :, i] for i in range(min(p, max_lag))]
                  
                  # Compute bootstrap IRFs
                  boot_psi = [np.eye(K)]
                  for h in range(1, horizon + 1):
                      temp = np.zeros((K, K))
                      for i in range(min(p, max_lag)):
                          if h - (i + 1) >= 0 and i < len(boot_Phi):
                              temp += boot_Phi[i] @ boot_psi[h - (i + 1)]
                      if h <= q and h - 1 < len(Theta):
                          temp += Theta[h - 1]
                      boot_psi.append(temp)
                  
                  boot_Psi = np.array(boot_psi)
                  if orth:
                      boot_Sigma = np.cov(boot_resids.T)
                      try:
                          P = cholesky(boot_Sigma)
                          boot_irf = np.array([boot_Psi[i] @ P for i in range(horizon + 1)])
                      except np.linalg.LinAlgError:
                          logger.warning(f"Bootstrap iteration {b} failed: Non-positive definite covariance")
                          boot_irf = boot_Psi
                  else:
                      boot_irf = boot_Psi
                  
                  boot_irfs[b] = boot_irf
              except Exception as e:
                  logger.warning(f"Bootstrap iteration {b} failed: {e}")
                  boot_irfs[b] = irf
          
          ci_lower = np.percentile(boot_irfs, 100 * self.ci_alpha / 2, axis=0)
          ci_upper = np.percentile(boot_irfs, 100 * (1 - self.ci_alpha / 2), axis=0)
      
      if plot:
          fig, axes = plt.subplots(K, K, figsize=(12, 8), sharex=True)
          axes = np.array(axes).flatten() if K > 1 else [axes]
          for i in range(K):
              for j in range(K):
                  idx = i * K + j
                  axes[idx].plot(range(horizon + 1), irf[:, i, j], label=f'Shock {self.columns[j]} → {self.columns[i]}', color='blue')
                  if bootstrap:
                      axes[idx].fill_between(range(horizon + 1), ci_lower[:, i, j], ci_upper[:, i, j],
                                            alpha=0.3, color='red', label=f'{100 * (1 - self.ci_alpha)}% CI')
                  axes[idx].set_title(f'{self.columns[i]} response to {self.columns[j]} shock')
                  axes[idx].set_xlabel('Horizon')
                  axes[idx].set_ylabel('Response')
                  axes[idx].grid(True, alpha=0.3)
                  axes[idx].legend()
          plt.tight_layout()
          plt.show()
      
      return irf if not bootstrap else (irf, ci_lower, ci_upper)

  def fevd(self, horizon=10, plot=True):
      if self.best_model is None:
          raise ValueError("No model fitted. Cannot compute FEVD.")
      
      K = len(self.columns)
      irf = self.compute_irf(horizon=horizon, orth=True, bootstrap=False, plot=False)
      
      # Compute FEVD using orthogonalized IRFs
      fevd_matrix = np.zeros((horizon + 1, K, K))
      total_var = np.zeros((horizon + 1, K))
      for H in range(horizon + 1):
          for a in range(K):
              total = 0
              for b in range(K):
                  contrib = np.sum(irf[:H + 1, a, b] ** 2)
                  fevd_matrix[H, a, b] = contrib
                  total += contrib
              total_var[H, a] = total
          for a in range(K):
              if total_var[H, a] > 0:
                  fevd_matrix[H, a, :] /= total_var[H, a]
      
      if plot:
          fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
          axes = [axes] if K == 1 else axes
          for j in range(K):
              bottom = np.zeros(horizon + 1)
              for k in range(K):
                  axes[j].bar(range(horizon + 1), fevd_matrix[:, j, k] * 100, bottom=bottom,
                              label=f'Shock from {self.columns[k]}', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][k % 4])
                  bottom += fevd_matrix[:, j, k] * 100
              axes[j].set_title(f'FEVD for {self.columns[j]}')
              axes[j].set_xlabel('Horizon')
              axes[j].set_ylabel('Variance Contribution (%)')
              axes[j].legend()
              axes[j].grid(True, alpha=0.3)
          plt.tight_layout()
          plt.show()
      
      return fevd_matrix

  def predict(self, n_periods=6, plot=True, tol=1e-6):
      if self.best_model is None:
          raise ValueError("No model fitted. Cannot generate forecasts.")
      if not self.fitted:
          logger.warning("The model is not fully fitted; forecasts may be unreliable.")
      
      K = len(self.columns)
      p = self.best_p
      q = self.best_q
      max_lag = max(p, q, 1)
      T = len(self.data)
      
      # Extract AR and MA coefficients
      if self.structural_id:
          AR_l0 = self.best_model.get('MA/AR_l0', np.eye(K))
          AR = self.best_model.get('AR', np.zeros((K, K * p)))
          MA = self.best_model.get('MA', np.zeros((K, K * q)))
          try:
              AR_l0_inv = inv(AR_l0)
          except np.linalg.LinAlgError:
              logger.warning("AR_l0 matrix is singular, using pseudoinverse")
              AR_l0_inv = np.linalg.pinv(AR_l0)
          Phi = [AR_l0_inv @ AR[:, i * K:(i + 1) * K] for i in range(p)] if p > 0 else []
          Theta = [AR_l0_inv @ MA[:, j * K:(j + 1) * K] for j in range(q)] if q > 0 else []
      else:
          AR = self.best_model.get('AR', np.zeros((K, K * p)))
          MA = self.best_model.get('MA', np.zeros((K, K * q)))
          Phi = [AR[:, i * K:(i + 1) * K] for i in range(p)] if p > 0 else []
          Theta = [MA[:, j * K:(j + 1) * K] for j in range(q)] if q > 0 else []
      
      # Initialize forecasts and data
      forecasts = np.zeros((n_periods, K))
      forecast_vars = np.zeros((n_periods, K))
      last_data = self.data.values[-max_lag:].copy()
      last_residuals = self.best_model['residuals'][-max_lag:].copy() if self.best_model['residuals'].shape[0] >= max_lag else np.zeros((max_lag, K))
      Sigma = np.cov(self.best_model['residuals'].T)
      # print(last_data)
      # print(Phi)
      # print(Theta)
      # Compute MA coefficients for variance (ψ_h)
      psi = [np.eye(K)]
      for h in range(1, n_periods + 1):
          temp = np.zeros((K, K))
          for i in range(min(p, len(Phi))):
              if h - (i + 1) >= 0:
                  temp += Phi[i] @ psi[h - (i + 1)]
          if h <= q and (h - 1) < len(Theta):
              temp += Theta[h - 1]
          psi.append(temp)
      # Forecast generation
      for h in range(n_periods):
          forecast_h = np.zeros(K)
          # AR contribution
          for i in range(p):
              if h - i - 1 >= 0:
                  # Use previously forecasted values
                  forecast_h += Phi[i] @ forecasts[h - i - 1]
              else:
                  # Use observed data
                  idx = max_lag - (i + 1 - h)
                  if 0 <= idx < len(last_data):
                      forecast_h += Phi[i] @ last_data[idx]
          # MA contribution
          for j in range(q):
              if h - j - 1 >= 0:
                  # Use forecasted residuals (usually 0, unless simulated)
                  forecast_h += 0  # or use stored forecasted residuals if available
              else:
                  idx = max_lag - (j + 1 - h)
                  if 0 <= idx < len(last_residuals):
                      forecast_h += Theta[j] @ last_residuals[idx]

          forecasts[h] = forecast_h
          # Forecast variance
          for k in range(K):
              var = 0
              for s in range(h + 1):
                  var += psi[s][k, :] @ Sigma @ psi[s][k, :].T
              forecast_vars[h, k] = var
      # Compute confidence intervals
      se = np.sqrt(forecast_vars)
      ci_lower = forecasts - norm.ppf(1 - self.ci_alpha / 2) * se
      ci_upper = forecasts + norm.ppf(1 - self.ci_alpha / 2) * se
      
      # Create forecast DataFrames
      if isinstance(self.data.index, pd.DatetimeIndex):
          # Infer frequency if not set
          freq = self.data.index.freq
          if freq is None:
              logger.warning("Data index frequency not set; inferring from data")
              time_diffs = np.diff(self.data.index)
              if len(time_diffs) > 0:
                  median_diff = pd.Timedelta(np.median(time_diffs))
                  if median_diff == pd.Timedelta(days=1):
                      freq = 'D'
                  elif median_diff == pd.Timedelta(weeks=1):
                      freq = 'W'
                  elif median_diff >= pd.Timedelta(days=28) and median_diff <= pd.Timedelta(days=31):
                      freq = 'M'
                  else:
                      freq = 'D'
                      logger.warning("Could not infer frequency; defaulting to daily")
              else:
                  freq = 'D'
          forecast_dates = pd.date_range(
              start=self.data.index[-1] + pd.Timedelta(days=1) if freq == 'D' else self.data.index[-1] + pd.offsets.MonthBegin(1) if freq == 'M' else self.data.index[-1] + pd.offsets.Week(1),
              periods=n_periods,
              freq=freq
          )
      else:
          forecast_dates = range(T, T + n_periods)
      
      forecast_df = pd.DataFrame(forecasts, index=forecast_dates, columns=self.columns)
      ci_lower_df = pd.DataFrame(ci_lower, index=forecast_dates, columns=self.columns)
      ci_upper_df = pd.DataFrame(ci_upper, index=forecast_dates, columns=self.columns)
      
      if plot:
          n_vars = len(self.columns)
          n_cols = min(2, n_vars)
          n_rows = (n_vars + n_cols - 1) // n_cols
          fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
          axes = np.array(axes).flatten() if n_vars > 1 else [axes]
          
          # Plot historical and fitted data
          fitted = self.best_model.get('fitted', None)
          train_data = self.data.iloc[max_lag:]
          fitted_df = pd.DataFrame(fitted, index=train_data.index, columns=self.columns) if fitted is not None and fitted.shape[0] == len(train_data) else None
          
          for i, col in enumerate(self.columns):
              ax = axes[i]
              hist_data = self.data[col].iloc[-min(50, len(self.data)):]
              ax.plot(hist_data.index, hist_data.values, 'b-', label='Historical', linewidth=1.5)
              if fitted_df is not None:
                  ax.plot(fitted_df.index, fitted_df[col], 'g--', label='Fitted', linewidth=1.5)
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
  def simulate(self, n_periods=100, plot=True, tol=1e-6):
      """
      Simulate time series data from the fitted VARMA model.

      Args:
          n_periods (int): Number of periods to simulate (default: 100).
          plot (bool): Whether to plot simulated series.
          tol (float): Tolerance for numerical stability.

      Returns:
          ndarray: Simulated series of shape (n_periods, K).

      Raises:
          ValueError: If model is not fitted, n_periods is invalid, or data issues are detected.
      """
      if self.best_model is None:
          raise ValueError("No model fitted. Cannot simulate.")

      # Input validation
      K = len(self.columns)
      if not isinstance(n_periods, int) or n_periods <= 0:
          raise ValueError(f"Number of periods must be a positive integer, got {n_periods}")
      if n_periods > 10000:
          logger.warning(f"Large number of periods ({n_periods}) may be computationally expensive")

      p = self.best_model['p']
      q = self.best_model['q']
      Ph0 = self.best_model['MA/AR_l0']
      PH = self.best_model['AR']
      TH = self.best_model['MA']
      residuals = self.best_model['residuals']
      data_mean = self.data.mean().values
      maxk = max(p, q)

      # Validate residuals
      T, K_resid = residuals.shape
      if K_resid != K:
          raise ValueError(f"Residuals have {K_resid} columns, expected {K}")
      if T < maxk:
          logger.warning(f"Sample size ({T}) is less than max(p, q) ({maxk}); simulation may be unreliable")

      # Compute AR and MA coefficients
      try:
          Ph0i = inv(Ph0) if self.structural_id else np.eye(K)
      except np.linalg.LinAlgError:
          raise ValueError("Cannot invert Ph0 matrix; check for singularity")
      ARc = Ph0i @ PH
      MAc = Ph0i @ TH

      # Initialize simulation
      Y_sim = np.zeros((n_periods + maxk, K))
      Y_sim[:maxk] = self.data.values[-maxk:] if len(self.data) >= maxk else np.zeros((maxk, K))
      at_sim = np.zeros((n_periods + maxk, K))
      at_sim[:maxk] = residuals[-maxk:] if T >= maxk else np.zeros((maxk, K))

      # Residual covariance
      Sigma = np.cov(residuals.T)
      Sigma += tol * np.eye(K)  # Ensure numerical stability
      try:
          np.linalg.cholesky(Sigma)
      except np.linalg.LinAlgError:
          logger.warning("Residual covariance matrix is not positive definite; adding perturbation")
          Sigma += tol * np.eye(K)

      # Simulate series
      for t in range(maxk, n_periods + maxk):
          Y_t = np.zeros(K)
          for j in range(1, min(p + 1, maxk + 1)):
              if j * K <= ARc.shape[1]:
                  Y_t += ARc[:, (j - 1) * K:j * K] @ Y_sim[t - j]
          for j in range(1, min(q + 1, maxk + 1)):
              if j * K <= MAc.shape[1]:
                  Y_t += MAc[:, (j - 1) * K:j * K] @ at_sim[t - j]
          Y_t += multivariate_normal.rvs(mean=np.zeros(K), cov=Sigma)
          Y_sim[t] = Y_t
          at_sim[t] = Y_t - (ARc[:, :min(p * K, ARc.shape[1])] @ Y_sim[t - min(p, maxk):t][::-1].flatten() if p > 0 else 0)

      Y_sim = Y_sim[maxk:] + data_mean  # Add mean back

      # Diagnostic check: Verify simulation sanity
      if np.any(np.isnan(Y_sim)) or np.any(np.isinf(Y_sim)):
          logger.warning("NaN or Inf values detected in simulated series; results may be unreliable")

      if plot:
          fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
          axes = [axes] if K == 1 else axes
          for i in range(K):
              axes[i].plot(Y_sim[:, i], label=f'Simulated {self.columns[i]}', color='r', linewidth=1.5)
              axes[i].set_title(f'Simulated Series for {self.columns[i]}')
              axes[i].set_xlabel('Time')
              axes[i].set_ylabel('Value')
              axes[i].legend()
              axes[i].grid(True, alpha=0.3)
          plt.tight_layout()
          plt.show()

      return Y_sim