from scipy.stats import chi2
from numpy.linalg import svd, pinv
from scipy.stats import t, norm
import numpy as np
import pandas as pd
import warnings
def ols_estimator(X,Y,tol=1e-6):
      #building The OlS estimator
  #I'll use numpy.linalg.lstsq ot reinviting the wheel
  #Fist check if X and Y aren't empty
  if X.size == 0 or Y.size == 0:
    raise ValueError("X or Y is empty")

  add_intercept=True
  #Fist as usual checking the inputs
  if X.shape[0] != Y.shape[0]:
    raise ValueError("X and Y don't share the same dimension")
  else:
    results={}
    ##Observations
    T=X.shape[0]
    ## Vars
    K=Y.shape[1]
    ###
    ###check the mean to see if an itrecept needs to be added
    col_means=np.mean(X,axis=0)
    demeaned_flags = np.isclose(col_means, 0, atol=tol)
    if np.all(demeaned_flags):
      X_full=X
      add_intercept=False
    else:
      add_intercept=True
      X_full=np.hstack((np.ones((T,1)),X))
    # Ensure Y is a NumPy array for consistent indexing
    Y_np = Y.to_numpy() if isinstance(Y, pd.DataFrame) else Y
    resid=np.zeros_like(Y_np)
    beta,_,rank,_=np.linalg.lstsq(X_full,Y,rcond=None)
    #print(beta)
    if add_intercept:
      beta_a = beta.copy()
      for j in range(K):
        if demeaned_flags[j]:
          beta_a[0,j] = 0
    else:
      beta_a=beta
    fitted=X_full@beta_a
    # Use NumPy array indexing for residue calculation
    resid=Y_np-fitted
    #### Diagnostics:
    #==============Rss and R2================#
    #Rss per var:
    rss_per_var = np.sum(resid**2, axis=0)
    tss_per_var = np.sum((Y_np-np.mean(Y_np, axis=0))**2, axis=0)
    #Rss_vec
    rss_vec=np.sum(rss_per_var)
    #TSS_vec
    tss_vec=np.sum(tss_per_var)
    ##Rsquare
    R_square_per_var=np.zeros(K)
    R_square=1-rss_vec/tss_vec if tss_vec > 0 else 0
    for i in range(K):
      R_square_per_var[i]=1-rss_per_var[i]/tss_per_var[i] if tss_per_var[i] > 0 else 0
    #=======
    #estimated_error_var
    dof=T-rank
    #print('ols, dof ',dof)
    ee_var=np.sqrt(rss_per_var/dof) if dof > 0 else np.zeros(K)
    #var_cov matrix of coeff
    var_beta=np.zeros(K)
    try:
      XTX_inv = np.linalg.inv(X_full.T @ X_full)
    except np.linalg.LinAlgError:
      XTX_inv = pinv(X_full.T @ X_full)
    #calculate diagonal of (XTX_inv * ee_var^2)
    se = np.sqrt(np.diag(XTX_inv).reshape(-1, 1) * (ee_var.reshape(1, -1)**2))
    z_values = np.where(se>0, beta_a / se, np.nan)
    if T < 30 and T > rank:
        p_values=2*(1-t.cdf(np.abs(z_values), df=T-rank))
    else:
        p_values= 2*(1-norm.cdf(np.abs(z_values)))
    #======Log-likelihood
    big_eps=resid.T@resid/T
    try:
        log_det_big_eps=np.log(np.linalg.det(big_eps + 1e-10 * np.eye(K)))
        log_lik = -0.5 * T * K * (np.log(2 * np.pi) + 1) - 0.5 * T * log_det_big_eps
    except np.linalg.LinAlgError:
        log_lik = -np.inf
    res={
    "resid": resid,
    "se": se,
    "z_values": z_values,
    "p_values": p_values,
    "R2": R_square,
    "R2_per_var": R_square_per_var,
    "log_likelihood": log_lik}
    return beta_a,fitted,resid,res