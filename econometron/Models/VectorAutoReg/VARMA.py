import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from econometron.Models.VectorAutoReg.VAR import VAR
from econometron.utils.estimation.Regression import ols_estimator
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.stattools import durbin_watson
from sklearn.cross_decomposition import CCA
from scipy.stats import chi2, norm, jarque_bera, shapiro, probplot, multivariate_normal
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, breaks_cusumolsresid
from numpy.linalg import inv, eigvals, det, cholesky, slogdet
from joblib import Parallel, delayed
import warnings
import logging
from scipy.optimize import minimize, approx_fprime
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VARMA(VAR):
    def __init__(self, data, max_p=5, max_q=5, columns=None, forecast_h=6, plot=True, check_stationarity=True, bootstrap_n=1000, criterion='AIC', structural_id=False, ci_alpha=0.05, Key=None, Threshold=0.8, orth=False, enforce_stab_inver=False):
        super().__init__(data, max_p, columns, criterion, forecast_h, plot,
                         bootstrap_n, ci_alpha, orth, check_stationarity, None, Threshold)
        self.max_q = max_q
        self.structural_id = structural_id
        self.best_model = None
        self.fitted = False
        self.columns = self.data.columns
        self.key = Key
        #####
        self.Kronind = None
        self.best_model = None
        self.best_q = None
        self.best_p = None
        self.AR_s = None
        self.MA_s = None
        self.coeff_table = pd.DataFrame()
        self.Threshold = Threshold
        self.forecast_h = forecast_h
        self.stab_inver = enforce_stab_inver
        self.best_model = {}
        self.Ph0 = None
        # self.indexes=None
    ###########

    def kron_index(self, lag):
        T, K = self.data.shape
        start_f = T-lag
        past = self.data.iloc[:start_f, :].to_numpy()
        p_r_ts = np.zeros((start_f, K*lag))
        p_r_ts[:, K*(lag-1):] = past
        for i in range(1, lag):
            p_i = self.data.iloc[i:i + start_f, :].to_numpy()
            p_r_ts[:, K * (lag - (i + 1)):K * (lag - i)] = p_i
        kdx = np.zeros(K, dtype=int)
        found = np.zeros(K, dtype=int)
        cstar = []
        star = lag
        h = 0
        futu = None
        while sum(found) < K:
            past = p_r_ts[:T-lag-h, :]
            # print(past.shape)
            futu1 = self.data.iloc[star + h:T, :].to_numpy()
            # print(futu1.shape)
            if futu is not None and futu.shape[0] > past.shape[0]:
                futu = futu[:past.shape[0], :]
            for i in range(K):
                if found[i] == 0:
                    if h == 0:
                        s1 = [j for j in range(
                            K) if found[j] == 0 and j < i]+[i]
                        futu = futu1[:, s1]
                    else:
                        futu = np.column_stack(
                            (futu, futu1[:, i])) if futu is not None else futu1[:, i:i+1]
                    n = min(past.shape[1], futu.shape[1])
                    cca = CCA(n_components=n, scale=True)
                    X_c, Y_c = cca.fit_transform(past, futu)
                    corr = [np.corrcoef(X_c[:, j], Y_c[:, j])[0, 1]
                            for j in range(X_c.shape[1])]
                    dp, df = past.shape[1], futu.shape[1]
                    # print(df)
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
                    tst = -(n-0.5*(dp+df-1))*np.log(1-sccs/dsq)
                    pv = 1-chi2.cdf(tst, deg)
                    stat = [tst, deg, pv]+([dsq] if h > 0 else [])
                    cstar.append(stat)
                    print(
                        f"Component {i + 1}: sccs={sccs:.6f}, tst={tst:.3f}, deg={deg}, pv={pv:.3f}, dsq={dsq:.3f}")
                    if pv > self.ci_alpha:
                        found[i] = 1
                        kdx[i] = h
                        print(
                            f"Component {i + 1}: Kronecker index {h}, pv={pv:.3f}")
                        if h > 0:
                            futu = futu[:, :df-1]
            h += 1
        return {"index": kdx, "tests": cstar}
    ######### identify#####################

    def struct_id(self, ord=None, use_var=False, output=True):
        T = self.data.shape[0]
        order = 0
        if ord is None and not use_var or use_var:
            logger.warning("Using VAR to identify structural parameters")
            max_theoretical_p = int(np.sqrt(T))
            candidate_max_p = min(self.max_p, max_theoretical_p)
            if candidate_max_p > 1:
                var = super().fit(p=candidate_max_p, output=False)
                order = var['p']
            else:
                order = T // K**2
                logger.warning(f"Order is {order}")
        else:
            if ord and not use_var:
                order = ord
        kd = self.Kronind = self.kron_index(order)['index']
        K = len(kd)
        idx = np.argsort(kd)
        self.best_p = np.max(kd)
        self.best_q = self.best_p
        mx = (self.best_p+1)*K
        MA = np.full((K, mx), 2, dtype=int)
        for i in range(K):
            MA[i, i] = 1
            if kd[i] < self.best_q:
                j = (kd[i]+1)*K
                MA[i, j:mx] = 0
        if K > 1:
            for i in range(K-1):
                MA[i, i+1:K] = 0
        AR = MA.copy()
        if K > 1:
            for i in range(1, K):
                for j in range(i):
                    if kd[j] <= kd[i]:
                        AR[i, j] = 0
        MA[:, :K] = AR[:, :K]
        for i in range(K):
            for j in range(K):
                if kd[i] > kd[j]:
                    for n in range(1, kd[i]-kd[j]+1):
                        AR[i, (n*K)+j] = 0
        if output:
            print("AR coefficient matrix:")
            print(AR)
            print("MA coefficient matrix:")
            print(MA)
        return {"AR_s_id": AR, "MA_s_id": MA}
    ########################################

    def _ini_s1(self, ord=None, output=True, p=None, q=None):
        T, K = self.data.shape
        estims = []
        s_e = []
        if T > 20:
            max_pq = min(10, int(round(np.sqrt(T))))
            print('1', max_pq)
        else:
            max_pq = 1
        if ord is None:
            if self.max_p > 1 or self.max_q > 1:
                # print('max_p',self.max_p)
                # print('max_q',self.max_q)
                cand = max(self.max_p, self.max_q) + 7
                # print('2',cand)
                max_pq = cand if cand < T else min(
                    int(round(np.sqrt(T))), max(1, int(T // (K**2))))
                # print('3',max_pq)

        else:
            if ord > 1:
                max_pq = ord
                print('ord', ord)
        # print('maaaaaaaaaaaaaaaaaax',max_pq)
        Hov = VAR(data=self.data, max_p=max_pq).fit(output=False)
        resids = Hov['residuals']
        p_v = Hov['p']
        # print('Var Order',p_v)
        _, Y = super().lag_matrix(p_v)
        X = np.array(resids)
        Y = np.array(Y)
        T1, K1 = Y.shape
        # print("w-----shapes------w")
        # print("this data", Y.shape)
        # print("this res", X.shape)
        b = None
        if self.structural_id:
            struct_matx = self.struct_id(ord=p_v, output=output)
            AR = self.AR_s = struct_matx['AR_s_id']
            MA = self.MA_s = struct_matx['MA_s_id']
            p_q = max(int(np.floor(AR.shape[1] / K) - 1), 1)
            for i in range(K):
                Y_i = Y[p_q:T1, i].reshape(-1, 1)
                X_list = [np.ones((Y_i.shape[0], 1))]
                icnt = 1
                if i > 0:
                    for j in range(i):
                        if AR[i, j] > 1:
                            tmp = (X[p_q:T1, j] - Y[p_q:T1, j]).reshape(-1, 1)
                            X_list.append(tmp)
                            icnt += 1
                for l in range(1, p_q + 1):
                    j_ = l * K
                    for j in range(K):
                        idx = j_ + j
                        if AR[i, idx] > 1:
                            tmp = Y[p_q - l:T1 - l, j].reshape(-1, 1)
                            X_list.append(tmp)
                            icnt += 1

                for ll in range(1, p_q + 1):
                    j_ = ll * K
                    for j in range(K):
                        idx = j_ + j
                        if MA[i, idx] > 1:
                            tmp = X[p_q - ll:T1 - ll, j].reshape(-1, 1)
                            X_list.append(tmp)
                            icnt += 1
                X_matrix = np.hstack(X_list) if icnt > 0 else np.ones(
                    (Y_i.shape[0], 1))
                beta_a, _, _, diag = ols_estimator(
                    X_matrix, Y_i, add_intercept=False)
                estims.extend(beta_a.flatten())
                s_e.extend(diag['se'])
                # if i == 0:
                #     new_w = X_matrix  # save for return
        else:
            ist = max(p or 0, q or 0)
            Y_ = Y[ist:, :]
            X_ = []
            # print(Y_.shape)
            # print(X.shape)
            # print(p)
            if p:
                # print('ok',p)
                for j in range(1, p+1):
                    tmp = Y[ist-j:T1-j, :]
                    # print(tmp)
                    X_.append(tmp)
            if q:
                for j in range(1, q+1):
                    tmp = X[ist-j:T1-j, :]
                    # print(tmp)
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
    def _prepare_for_est(self, estimates, stand_err, output=True):
        # print('eeee',estimates)
        if estimates is None or stand_err is None:
            raise ValueError(
                'estimates are None , cannot proceed with estimation')
        par = np.array([])
        separ = np.array([])
        if len(estimates) == 0:
            return par, separ, par, separ
        if not self.structural_id:
            est_vec = estimates.flatten(order='F')
            se_vec = stand_err.flatten(order='F')
            fixed = np.ones_like(estimates)
            mask = fixed.flatten(order='F') == 1
        else:
            beta_flat = np.concatenate([b.flatten() for b in estimates]) if isinstance(
                estimates, list) else estimates.flatten()
            se_flat = np.concatenate([s.flatten() for s in stand_err]) if isinstance(
                stand_err, list) else stand_err.flatten()
            est_vec = beta_flat
            se_vec = se_flat
            # All parameters should be estimated
            fixed = np.ones(len(est_vec), dtype=int)
            mask = fixed == 1
        par = est_vec[mask]
        separ = se_vec[mask]
        # Calculate bounds
        lowerBounds = par - 2 * separ
        upperBounds = par + 2 * separ
        if output:
            print(f"Number of parameters: {len(par)}")
            print("-" * 55)
            print(
                f"{'ini_est':^12} | {'lower bound':^12} | {'upperbound':^12} | {'s.err':^12}")
            print("-" * 55)
            for i in range(len(par)):
                ini = np.round(par[i], 4)
                lb = np.round(lowerBounds[i], 4)
                ub = np.round(upperBounds[i], 4)
                se = np.round(separ[i], 4)
                print(f"{ini:^12} | {lb:^12} | {ub:^12} | {se:^12}")
        return par, separ, lowerBounds, upperBounds
        # prepare AR and MA matrices :

    def prepare_A_B_Matrices(self, par, p=None, q=None):
        T, K = self.data.shape
        Kpar = par.copy()
        Cst = np.zeros((K, 1))
        indexes = []
        if self.structural_id:
            ARid = self.AR_s.copy()
            MAid = self.MA_s.copy()
            K = ARid.shape[0]
            kp1 = ARid.shape[1]
            kp = kp1 - K
            Ph0 = np.eye(K)
            A = np.zeros((K, kp))
            B = np.zeros((K, kp))
            icnt = 0
            for i in range(K):
                idx = np.where(ARid[i, :] > 1)[0]
                jdx = np.where(MAid[i, :] > 1)[0]
                kdx = np.where(ARid[i, :K] > 1)[0]
                if len(kdx) > 0:
                    mask_idx = ~np.isin(idx, kdx)
                    mask_jdx = ~np.isin(jdx, kdx)
                    idx = idx[mask_idx]
                    jdx = jdx[mask_jdx]
                iend = len(idx)
                jend = len(jdx)
                kend = len(kdx)
                # Constant term
                Cst[i, 0] = Kpar[icnt]
                indexes.append((icnt, i))
                icnt += 1
                if kend > 0:
                    for k in kdx:
                        Ph0[i, k] = Kpar[icnt]
                        indexes.append((icnt, K + k))
                        icnt += 1
                # AR matrix (A)
                if iend > 0:
                    lag_idx = idx - K
                    for lag in lag_idx:
                        j = lag % K
                        A[i, lag] = Kpar[icnt]
                        indexes.append((icnt, K + K * (lag // K) + i * K + j))
                        icnt += 1
                # MA matrix (B)
                if jend > 0:
                    lag_jdx = jdx - K
                    for lag in lag_jdx:
                        j = lag % K
                        B[i, lag] = Kpar[icnt]
                        indexes.append(
                            (icnt, K + kp + K * (lag // K) + i * K + j))
                        icnt += 1
            assert icnt == len(
                Kpar), f"Only used {icnt} of {len(Kpar)} parameters."
            Ph0i = inv(Ph0 + 1e-6 * np.eye(K))
            A = Ph0i @ A
            B = Ph0i @ B
            Cst = Ph0i @ Cst
            p = q = self.Kronind.max()
            ARs = A.reshape(p, K, K)
            MAs = B.reshape(q, K, K)
            self.Ph0 = Ph0
            return ARs, MAs, Cst.T, indexes
        else:
            dist = int(len(Kpar) / K)
            for i in range(K):
                Cst[i, 0] = Kpar[i * dist]
                indexes.append((i * dist, i))  # j = i for Cst[i]
            idx_l = [i * dist for i in range(K)]
            Kpar = np.delete(Kpar.copy(), idx_l)
            kp = K ** 2 * p if p is not None else 0
            kq = K ** 2 * q if q is not None else 0
            A = Kpar[:kp].reshape(K, p * K) if p > 0 else np.zeros((K, 0))
            B = Kpar[kp:kp + kq].reshape(K, q *
                                         K) if q > 0 else np.zeros((K, 0))
            A = A.reshape(p, K, K)
            B = B.reshape(q, K, K)
            icnt = len(idx_l)
            for lag in range(p):
                for i in range(K):
                    for j in range(K):
                        # j encodes A[lag, i, j]
                        indexes.append((icnt, K + lag * K * K + i * K + j))
                        icnt += 1
            for lag in range(q):
                for i in range(K):
                    for j in range(K):
                        # j encodes B[lag, i, j]
                        indexes.append(
                            (icnt, K + p * K * K + lag * K * K + i * K + j))
                        icnt += 1
            assert icnt == len(
                Kpar) + len(idx_l), f"Only used {icnt} of {len(Kpar) + len(idx_l)} parameters."
            return A, B, Cst.T, indexes

    def LL_func(self, par, p=None, q=None, verbose=False):
        ARs, MAs, Cst, indexes = self.prepare_A_B_Matrices(par, p=p, q=q)
        # print("shapes",ARs.shape,MAs.shape)
        T, K = self.data.shape
        data = np.array(self.data)
        ar_ = p if p is not None else self.Kronind.max()
        ma_ = q if q is not None else self.Kronind.max()
        slag = max(ar_, ma_)
        ####
        # The Varma model is Y=C+AR*Y+MA*eps +Big_eps
        # suppose
        Y = self.data.to_numpy().T
        Eps = np.zeros((K, T))
        resid = np.zeros((K, T))
        for t in range(T):
            ar_lags = ar_ if t >= ar_ else t
            ma_lags = ma_ if t >= ma_ else t
            varma_process = Cst.copy()
            if ar_lags:
                varma_process += sum(ARs[i, :, :]@Y[:, t-i-1]
                                     for i in range(ar_lags))
            if ma_lags:
                varma_process += sum(MAs[j, :, :]@Eps[:, t-j-1]
                                     for j in range(ma_lags))
            big_eps = Y[:, t] - varma_process
            # print('VARMA',varma_process)
            Eps[:, t] = big_eps
            resid[:, t] = big_eps
        resid = resid[:, slag:]
        reg = 1e-6*np.eye(K)
        Sigma = np.cov(resid, bias=True)
        try:
            inv_Sigma = np.linalg.inv(Sigma)
            sign, logdet = np.linalg.slogdet(Sigma)
        except np.linalg.LinAlgError:
            inv_Sigma = np.linalg.pinv(Sigma)
            sign, logdet = np.linalg.slogdet(Sigma)
            if verbose:
                print("Sigma singular -> penalty")
            return 1e7
        eff_t = resid.shape[1]
        ll = -(eff_t*K)/2*np.log(2*np.pi)-(eff_t)/2*logdet - \
            0.5*np.trace(resid.T @ inv_Sigma @ resid)
        neg_ll = -ll
        return neg_ll

    def _build_companion(self, matrix, K, order):
        if order == 0:
            return np.zeros((K, K))
        if not isinstance(matrix, np.ndarray) or matrix.shape != (order, K, K):
            raise ValueError(
                f"Expected matrix shape ({order}, {K}, {K}), got {matrix.shape}")
        comp_dim = K * order
        top = np.zeros((K, comp_dim))
        for i in range(order):
            top[:, i * K:(i + 1) * K] = matrix[i, :, :]
        if order == 1:
            return top
        else:
            bottom = np.eye(K * (order - 1), K *
                            order) if order > 1 else np.zeros((0, comp_dim))
            companion = np.vstack([top, bottom])
            if companion.shape[0] != companion.shape[1]:
                raise ValueError(
                    f"Companion matrix is not square: shape {companion.shape}")
            return companion

    def numerical_hessian(self, par, p, q, epsilon=1e-5):
        def neg_ll(x):
            return self.LL_func(x, p, q)
        n = len(par)
        hess = np.zeros((n, n))
        grad0 = approx_fprime(par, neg_ll, epsilon)
        for i in range(n):
            x1 = par.copy()
            x1[i] += epsilon
            grad1 = approx_fprime(x1, neg_ll, epsilon)
            hess[i, :] = (grad1 - grad0) / epsilon
        hess = (hess + hess.T) / 2
        return hess

    def compute_diags(self, gs=False, model=None):
        if gs:
            if model is None:
                raise ValueError(
                    "For grid search mode, model must be provided.")
            target = model
        else:
            if self.best_model is None:
                warnings.warn("No self_model is fitted")
                return
            target = self.best_model
        par = target['par']
        hess = self.numerical_hessian(par, target['p'], target['q'])
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            hess_inv = np.eye(len(par)) * 1e-6
        if hasattr(hess_inv, 'todense'):
            hess_inv = hess_inv.todense()
        se = np.sqrt(np.diag(hess_inv))
        t_vals = par / se
        p_vals = 2 * (1 - norm.cdf(np.abs(t_vals)))
        signif = []
        for pval in p_vals:
            if pval < 0.01:
                signif.append('***')
            elif pval < 0.05:
                signif.append('**')
            elif pval < 0.1:
                signif.append('*')
            else:
                signif.append('')
        diags = {'se': se, 'tvals': t_vals, 'pvals': p_vals, 'signif': signif}
        target.update(diags)
        if len(se) != len(par):
            print(
                f"Warning: Diagnostics length ({len(se)}) does not match parameters ({len(par)})")
        return diags

    def get_resids(self, A, B, Cst):
        p = A.shape[0] if A.size else 0
        q = B.shape[0] if B.size else 0
        Y = self.data.to_numpy().T
        K, T = Y.shape
        Eps = np.zeros((K, T))
        for t in range(T):
            pred = Cst.copy()
            for i in range(1, min(p, t) + 1):
                pred += A[i - 1] @ Y[:, t - i]
            for j in range(1, min(q, t) + 1):
                pred += B[j - 1] @ Eps[:, t - j]
            Eps[:, t] = Y[:, t] - pred
        self.best_model['residuals'] = Eps
        return Eps

    def get_crit(self, value, num_params, crit_name):
        T, k = self.data.shape
        sig = np.cov(value, bias=True)
        sig = np.linalg.det(sig)
        sig_ = np.log(sig)
        if value is not None or value != np.inf:
            if crit_name == 'aic':
                return sig_+(2*num_params)/T
            elif crit_name == 'bic':
                return sig_+np.log(T)*num_params/T
            elif crit_name == 'hqic':
                return sig_+2*np.log(np.log(T))*num_params/T if T > 2 else float('inf')
        else:
            return None

    def display_results(self, results):
        params = np.array(results.get('par', []))
        p = results.get('p')
        q = results.get('q')
        resid = results.get('residuals', np.nan)
        loglik = results.get('loglikelihood', np.nan)
        T, K = self.data.shape
        num_params = len(params)
        indexes = results.get('Index', [])
        # Criteria
        aic = self.get_crit(resid, num_params, 'aic')
        bic = self.get_crit(resid, num_params, 'bic')
        hqic = self.get_crit(resid, num_params, 'hqic')
        model_type = "Structural" if self.structural_id else "Non-Structural"
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Diagnostics
        se = np.array(results.get('se', np.full(num_params, np.nan)))
        tvals = np.array(results.get('tvals', np.full(num_params, np.nan)))
        pvals = np.array(results.get('pvals', np.full(num_params, np.nan)))
        signif = results.get('signif', [''] * num_params)
        # === HEADER ===
        print("=" * 70)
        print(
            f"{'Structural' if self.structural_id else 'Non-Structural'} VARMA({p},{q})".center(70))
        print("=" * 70)
        print(
            f"Log-likelihood: {loglik:.4f} | Model type: {model_type} | Time: {time_now}")
        print(
            f"AIC: {aic:.4f} | BIC: {bic:.4f} | HQIC: {hqic:.4f} | Hannan-Rissanen Method")
        print("=" * 70)
        # === Structural contemporaneous matrix ===
        if self.structural_id and hasattr(self, "Ph0"):
            print("\nContemporaneous Impact Matrix (Ph0):")
            if np.any(np.abs(self.Ph0 - np.eye(K)) > 1e-10):
                df_ph0 = pd.DataFrame(self.Ph0, columns=[
                                      f"Y{j+1}" for j in range(K)], index=[f"Y{i+1}" for i in range(K)])
                print(df_ph0.to_string())
            else:
                print("Identity matrix (no contemporaneous effects)")
            print("=" * 70)
        # === Parameter list ===
        rows = []
        indexes = indexes.copy()
        for idx, code in indexes:
            if code < K:
                # Constant
                lag, typ, src, dst = "", "Cst", "", f"Y{code+1}"
            elif code < K+p*K*K:
                # AR term
                rel = code - K
                lag = rel // (K*K) + 1
                rem = rel % (K*K)
                src = f"Y{rem // K + 1}"
                dst = f"Y{rem % K + 1}"
                typ = "AR"
            elif code < K + p*K*K + q*K*K:
                # MA term
                rel = code - (K + p*K*K)
                lag = rel // (K*K) + 1
                rem = rel % (K*K)
                src = f"Y{rem // K + 1}"
                dst = f"Y{rem % K + 1}"
                typ = "MA"
            else:
                # Ph0 structural param
                rel = code - (K + p*K*K + q*K*K)
                src = f"Y{rel // K + 1}"
                dst = f"Y{rel % K + 1}"
                lag, typ = "", "Ph0"
            rows.append({
                "Lag": lag,
                "Type": typ,
                "From": src,
                "To": dst,
                "Value": params[idx],
                "Std. Error": se[idx],
                "t-Stat": tvals[idx],
                "p-Value": pvals[idx],
                "Signif": signif[idx]
            })
        df = pd.DataFrame(rows)
        print("\nParameter Estimates:\n")
        print(df.to_string(index=False))
        print("=" * 70)
        return df

    def plot_fitted(self):
        if not self.best_model:
            return
        resids = self.best_model['residuals']
        data = self.data.to_numpy()
        fitted = data - resids.T
        for i, col in enumerate(self.columns):
            plt.figure()
            plt.plot(self.data.index, data[:, i], label='Actual')
            plt.plot(self.data.index, fitted[:, i], label='Fitted')
            plt.legend()
            plt.title(f"Fitted vs Actual for {col}")
            plt.show()

    def fit(self, p=None, q=None, plot=True, verbose=True, enforce_stab_inver=False):
        grid_search = False
        ini_p = p
        ini_q = q
        if verbose:
            logger.info(f"[INFO] Starting VARMA fit with p={p}, q={q}, "
                        f"stability={enforce_stab_inver}, plot={plot}")
        if self.structural_id:
            if verbose:
                logger.info("[INFO] Structural identification mode")
            ini_p, ini_q = None, None
            # self.struct_id(ord=None, use_var=True, output=verbose)
            # p = self.best_p
            # q = self.best_q
            grid_search = False
        else:
            if p is None or q is None:
                grid_search = True
                if verbose:
                    logger.info(
                        f"[INFO] Falling back to grid search over p=0 to {self.max_p}, q=0 to {self.max_q}")
            else:
                if verbose:
                    logger.info(
                        f"[INFO] Using non-structural VARMA, p={p}, q={q}")
        T, K = self.data.shape
        crit_name = self.criterion.lower()
        if grid_search:
            results = []
            pq_pairs = [(i, j) for i in range(self.max_p + 1)
                        for j in range(self.max_q + 1)]
            for pp, qq in pq_pairs:
                if pp == 0 and qq == 0:
                    model = {'p': 0, 'q': 0, 'par': np.array([0]), 'A': np.array([0]), 'B': np.array([0]), 'Cst': np.array([0]), 'residuals': np.array([0]), 'loglikelihood': 9999,
                             'Index': [], crit_name: 9999}
                    results.append(model)
                    continue
                try:
                    estimates, stand_err = self._ini_s1(
                        p=pp, q=qq, output=False)
                    if len(estimates) == 0:
                        continue
                    par, separ, lower_bounds, upper_bounds = self._prepare_for_est(
                        estimates, stand_err, output=False)
                    if len(par) == 0:
                        continue
                    constraints = []
                    if enforce_stab_inver:
                        constraints = [{
                            'type': 'ineq',
                            'fun': lambda x: 1.0 - np.max(np.abs(eigvals(self._build_companion(self.prepare_A_B_Matrices(x, pp, qq)[0], K, pp)))) - 1e-6 if pp > 0 else 1.0
                        },
                            {
                                'type': 'ineq',
                                'fun': lambda x: 1.0 - np.max(np.abs(eigvals(self._build_companion(self.prepare_A_B_Matrices(x, pp, qq)[1], K, qq)))) - 1e-6 if qq > 0 else 1.0
                        }
                        ]
                    result = minimize(self.LL_func, par, args=(pp, qq, False), method='trust-constr', bounds=list(zip(
                        lower_bounds, upper_bounds)), constraints=constraints, options={'disp': False, 'maxiter': 500, 'gtol': 1e-6, 'xtol': 1e-6})
                    if not result.success or np.isnan(result.fun) or np.isinf(result.fun):
                        if verbose:
                            logger.warning(
                                f"[WARN] Optimization failed for p={pp}, q={qq}: fun={result.fun}")
                        continue
                    A, B, Cst, indexes = self.prepare_A_B_Matrices(
                        result.x, p=pp, q=qq)
                    resids = self.get_resids(A, B, Cst)
                    cr = self.get_crit(resids, len(result.x), crit_name)
                    if np.isnan(cr) or np.isinf(cr):
                        continue
                    model = {'par': result.x, 'A': A, 'B': B, 'p': pp, 'q': qq, 'Cst': Cst,
                             crit_name: cr, 'residuals': resids, 'loglikelihood': -result.fun, 'Index': indexes}
                    results.append(model)
                except Exception as e:
                    if verbose:
                        logger.warning(
                            f"[WARN] Exception for p={pp}, q={qq}: {str(e)}")
                    continue
            if not results:
                raise ValueError("No valid models found during grid search.")
            # print(results)
            best_model = min(results, key=lambda m: m[crit_name])
            self.best_model = best_model
            self.best_p = best_model['p']
            self.best_q = best_model['q']
        else:
            try:
                if ini_p == 0 and ini_q == 0:
                    model = {'p': 0, 'q': 0, 'par': np.array([0]), 'A': np.array([0]), 'B': np.array([0]), 'Cst': np.array([0]), 'residuals': np.array([0]), 'loglikelihood': 9999,
                             'Index': [], crit_name: 9999}
                    return model
                else:
                    estimates, stand_err = self._ini_s1(
                        ord=None, output=verbose, p=ini_p, q=ini_q)
                    par, separ, lower_bounds, upper_bounds = self._prepare_for_est(
                        estimates, stand_err, output=verbose)
                    constraints = []
                    if enforce_stab_inver:
                        constraints = [{
                            'type': 'ineq',
                            'fun': lambda x: 1.0 - np.max(np.abs(eigvals(self._build_companion(self.prepare_A_B_Matrices(x, p, q)[0], K, p)))) - 1e-6 if p > 0 else 1.0
                        },
                            {
                                'type': 'ineq',
                                'fun': lambda x: 1.0 - np.max(np.abs(eigvals(self._build_companion(self.prepare_A_B_Matrices(x, p, q)[1], K, q)))) - 1e-6 if q > 0 else 1.0
                        }
                        ]
                    result = minimize(self.LL_func, par, args=(ini_p, ini_q, verbose), method='trust-constr', bounds=list(zip(lower_bounds, upper_bounds)), constraints=constraints,
                                      options={'disp': verbose, 'maxiter': 1000, 'gtol': 1e-6, 'xtol': 1e-6})
                    if not result.success or np.isnan(result.fun) or np.isinf(result.fun):
                        raise ValueError(
                            f"Optimization failed: {result.message}, fun={result.fun}")
                    A, B, Cst, indexes = self.prepare_A_B_Matrices(
                        result.x, p=ini_p, q=ini_q)
                    resids = self.get_resids(A, B, Cst)
                    cr = self.get_crit(resids, len(result.x), crit_name)
                    p = self.best_p if self.structural_id else ini_p
                    q = self.best_q if self.structural_id else ini_q
                    self.best_model = {'par': result.x, 'A': A, 'B': B, 'p': p, 'q': q, 'Cst': Cst,
                                       crit_name: cr, 'residuals': resids, 'loglikelihood': -result.fun, 'Index': indexes}
            except Exception as e:
                raise ValueError(f"Single model fit failed: {str(e)}")
        self.compute_diags()
        if verbose:
            print(
                f"Model fitted with p={self.best_model['p']}, q={self.best_model['q']}, Log-likelihood: {self.best_model['loglikelihood']:.4f}")
            self.display_results(self.best_model)
            print(f"{crit_name.upper()}: {self.best_model[crit_name]:.4f}")
            self.run_full_diagnosis()
            if self.fitted:
                logger.info("Model is fitted.")
            else:
                warnings.warn(
                    "The models is not well(Under/Over) Fitted on the Data")
        if plot:
            self.plot_fitted()
        return self.best_model

    def run_full_diagnosis(self, num_lags=8, plot=False, threshold=None):
        if threshold is None:
            threshold = getattr(self, 'Threshold', 0.8)
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold needs to be between 0 and 1")
        Diagnosis = {}
        self.columns = self.data.columns
        T = self.data.shape[0]
        K = len(self.columns)
        is_stable = False
        is_invertible = False
        if self.best_model is None:
            print("No model fitted. Cannot perform diagnostics.")
            Diagnosis['Final Diagnosis'] = 'Not Fitted'
            return Diagnosis
        resids = self.best_model.get('residuals')
        if resids.ndim == 2:
            if resids.shape[0] == K and resids.shape[1] != K:
                resids = resids.T
        if resids.ndim != 2 or resids.shape[1] != K:
            raise ValueError(
                f"Residuals shape mismatch. Expected ({T},{K}), got {resids.shape}")
        A = self.best_model.get('A', np.zeros((0, K, K)))
        B = self.best_model.get('B', np.zeros((0, K, K)))
        p = A.shape[0] if A.size else 0
        cm = self._build_companion(A, K, p)
        print("===================Stability===========================")
        if p > 0:
            max_modulus = np.max(np.abs(eigvals(cm)))
            print(f"Maximum eigenvalue modulus: {max_modulus:.4f}")
            print(
                f"The VARMA AR-part is {'stable' if is_stable else 'non-stable'}")
        else:
            is_stable = True
            print("No AR terms - automatically stable")
        Diagnosis['Final Stability Diagnosis'] = 'Stable' if is_stable else "Not Stable"
        S_score = 1.0 if is_stable else 0.0
        Diagnosis['Stability Score'] = S_score
        print("===================Invertibility===========================")
        B = self.best_model.get('B', np.zeros((0, K, K)))
        q = B.shape[0] if B.size else 0
        if q > 0:
            ma_comp_dim = K * q
            ma_top = np.zeros((K, ma_comp_dim))
            for i in range(q):
                ma_top[:, i * K:(i + 1) * K] = B[i, :, :]
            if ma_comp_dim == K:
                ma_companion = ma_top
            else:
                ma_bottom = np.eye(
                    K * (q - 1), K * q) if q > 1 else np.zeros((0, ma_comp_dim))
                ma_companion = np.vstack([ma_top, ma_bottom])
            ma_eigenvalues = np.linalg.eigvals(ma_companion)
            ma_moduli = np.abs(ma_eigenvalues)
            is_invertible = np.all(ma_moduli < 1.0)
            max_ma_modulus = np.max(ma_moduli)
            print(f"Maximum MA eigenvalue modulus: {max_ma_modulus:.4f}")
            print(
                f"The VARMA MA-part is {'invertible' if is_invertible else 'non-invertible'}")
        else:
            is_invertible = True
            print("No MA terms - automatically invertible")
        Diagnosis['Final Invertibility Diagnosis'] = 'Invertible' if is_invertible else "Not Invertible"
        I_score = 1.0 if is_invertible else 0.0
        Diagnosis['Invertibility Score'] = I_score
        both_conditions = is_stable and is_invertible
        print(
            f"\\nOverall Model Condition: {'Stationary and Invertible' if both_conditions else 'Conditions violated'}")
        Diagnosis['Model Condition'] = 'Stationary and Invertible' if both_conditions else 'Conditions violated'
        print("===================Serial COrrelation Tests===========================")
        lb_results = []
        dw_results = []
        print(f"===============Ljung–Box Test (lags={num_lags})==============")
        for i in range(K):
            try:
                lb_test = acorr_ljungbox(
                    resids[:, i], auto_lag=True, return_df=True)
                pval = lb_test['lb_pvalue'].values[0]
            except Exception:
                pval = 0.0
            LbT = "PASS" if pval > 0.05 else "FAIL"
            print(
                f"Residual {i} ({self.columns[i]}): p-value = {pval:.4f} → {LbT}")
            lb_results.append(LbT)
        print("==================Durbin-Watson Statistics=================")
        for i in range(K):
            dw = durbin_watson(resids[:, i])
            dw_result = "PASS" if 1.5 <= dw <= 2.5 else "FAIL"
            print(
                f"Residual {i} ({self.columns[i]}): DW = {dw:.4f} → {dw_result}")
            dw_results.append(dw_result)
        DW_score = dw_results.count('PASS') / K
        LB_score = lb_results.count('PASS') / K
        auto_corr_score = (DW_score + LB_score) / 2
        Diagnosis['DW_score'] = DW_score
        Diagnosis['LB_score'] = LB_score
        Diagnosis['Autocorrelation_score'] = auto_corr_score
        Diagnosis['DW_diagnosis'] = 'Passed' if DW_score == 1 else 'Failed'
        Diagnosis['LB_diagnosis'] = 'Passed' if LB_score == 1 else 'Failed'
        Diagnosis['Autocorrelation_diagnosis'] = 'Passed' if auto_corr_score == 1 else 'Failed'
        Diagnosis['Final autocorrelation Diagnosis'] = 'Passed' if DW_score == 1 and LB_score == 1 else 'Failed'
        print("==================Heteroscedasticity=================")
        arch_res = []
        for i in range(K):
            try:
                p_q = self.best_p+self.best_q
                arch_test = het_arch(resids[:, i], ddof=p_q)
                pval = arch_test[1]
            except Exception:
                pval = 0.0
            arch_result = 'PASS' if pval >= 0.05 else 'FAIL'
            arch_res.append(arch_result)
            print(
                f"Residual {i} ({self.columns[i]}): ARCH p-value = {pval:.4f} → {arch_result}")
        arch_score = arch_res.count('PASS') / K
        Diagnosis['Heteroscedasticity_score'] = arch_score
        Diagnosis['Heteroscedasticity_diagnosis'] = 'Passed' if arch_score == 1 else 'Failed'
        Diagnosis['Final Heteroscedasticity Diagnosis'] = 'Passed' if arch_score == 1 else 'Failed'
        print("=======================Normality Test=======================")
        jb_results = []
        shapiro_results = []
        for i in range(K):
            try:
                jb_test = jarque_bera(resids[:, i])
                jb_pval = jb_test.pvalue
            except Exception:
                jb_pval = 0.0
            try:
                if len(resids[:, i]) <= 5000:
                    sh_test = shapiro(resids[:, i])
                    sh_pval = sh_test.pvalue
                else:
                    from scipy.stats import anderson
                    ad_test = anderson(resids[:, i], dist='norm')
                    sh_pval = 0.05 if ad_test.statistic > ad_test.critical_values[2] else 0.1
            except Exception:
                sh_pval = 0.0
            print(
                f"Residual {i} ({self.columns[i]}): JB p-value = {jb_pval:.4f}, Shapiro p-value = {sh_pval:.4f}")
            jb_results.append('PASS' if jb_pval >= 0.05 else 'FAIL')
            shapiro_results.append('PASS' if sh_pval >= 0.05 else 'FAIL')
        jb_score = jb_results.count('PASS') / K
        shapiro_score = shapiro_results.count('PASS') / K
        normality_score = (jb_score + shapiro_score) / 2
        Diagnosis['JB_score'] = jb_score
        Diagnosis['Shapiro_score'] = shapiro_score
        Diagnosis['Normality_score'] = normality_score
        Diagnosis['Normality_diagnosis'] = 'Passed' if normality_score >= 0.5 else 'Failed'
        Diagnosis['Final Normality Diagnosis'] = 'Passed' if normality_score >= 0.5 else 'Failed'
        print("========================Structural Breaks============================")
        try:
            cusum_stat, cusum_pval, _ = breaks_cusumolsresid(resids, ddof=0)
        except Exception:
            cusum_pval = 0.0
        print(f"CUSUM p-value: {cusum_pval:.4f}")
        structural_breaks_pass = cusum_pval >= 0.05
        structural_breaks_score = 1.0 if structural_breaks_pass else 0.0
        Diagnosis['Structural_breaks_score'] = structural_breaks_score
        Diagnosis['Final Structural Breaks'] = 'Passed' if structural_breaks_pass else 'Failed'
        final_score = (S_score + I_score+auto_corr_score +
                       arch_score + normality_score+structural_breaks_score)/5
        self.fitted = final_score >= threshold
        Diagnosis['Final_score'] = final_score
        Diagnosis['Verdict'] = 'Passed' if self.fitted else 'Failed'
        print("\\n==================Diagnostic Summary=================")
        q = self.best_model.get('B').shape[0] if self.best_model.get(
            'B') is not None else 0
        summary_table = {
            'Estimation': 'Cond.MLE' if 'loglikelihood' in self.best_model else 'Numerical',
            'Model': f'VARMA({p},{q})',
            'Log-Likelihood': self.best_model.get('loglikelihood', 'N/A'),
            'AIC': self.best_model.get('aic', 'N/A'),
            'BIC': self.best_model.get('bic', 'N/A'),
            'Stability': f"{S_score:.3f} ({Diagnosis['Final Stability Diagnosis']})",
            'Autocorrelation': f"{auto_corr_score:.3f} ({Diagnosis['Autocorrelation_diagnosis']})",
            '  - DW Score': f"{DW_score:.3f} ({Diagnosis['DW_diagnosis']})",
            '  - LB Score': f"{LB_score:.3f} ({Diagnosis['LB_diagnosis']})",
            'Heteroscedasticity': f"{arch_score:.3f} ({Diagnosis['Heteroscedasticity_diagnosis']})",
            'Normality': f"{normality_score:.3f} ({Diagnosis['Normality_diagnosis']})",
            '  - JB Score': f"{jb_score:.3f}",
            '  - Shapiro Score': f"{shapiro_score:.3f}",
            'Structural Breaks': f"{structural_breaks_score:.3f} ({Diagnosis['Final Structural Breaks']})",
            'Final Score': f"{final_score:.3f}",
            'Verdict': f"{Diagnosis['Verdict']} (Threshold: {threshold})"}
        for key, value in summary_table.items():
            print(f"{key:<25} | {value}")
        print("-" * 50)
        if plot:
            T, _ = resids.shape
            fig_height = 4 * (K + 1)
            fig, axes = plt.subplots(
                nrows=K + 1, ncols=2, figsize=(12, fig_height))
            try:
                cusum_stat, cusum_pval, cusum_crit = breaks_cusumolsresid(
                    resids, ddof=0)
            except Exception:
                cusum_pval = 0.0
                cusum_crit = None
            flat_resid = resids.flatten()
            n = len(flat_resid)
            resid_centered = flat_resid - np.mean(flat_resid)
            resid_std = np.std(flat_resid, ddof=0)
            cusum_series = np.cumsum(resid_centered) / (resid_std * np.sqrt(n))
            critical_value = 1.36
            if cusum_crit is not None:
                for sig_level, crit_val in cusum_crit:
                    if sig_level == 5:
                        critical_value = crit_val
                        break
            ax_cusum = plt.subplot2grid((K + 1, 2), (0, 0), colspan=2)
            ax_cusum.plot(
                cusum_series, label='Standardized CUSUM of Residuals', color='blue')
            ax_cusum.axhline(y=critical_value, color='red', linestyle='--',
                             label=f'+Critical Value ({critical_value:.3f})')
            ax_cusum.axhline(y=-critical_value, color='red', linestyle='--',
                             label=f'-Critical Value ({-critical_value:.3f})')
            ax_cusum.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            test_result = "PASSED" if cusum_pval >= 0.05 else "FAILED"
            ax_cusum.set_title(
                f"CUSUM Test for Structural Breaks (p-value: {cusum_pval:.4f}, {test_result})")
            ax_cusum.set_xlabel("Time Index")
            ax_cusum.set_ylabel("Standardized CUSUM")
            ax_cusum.legend()
            ax_cusum.grid(True, alpha=0.3)
            for i in range(K):
                ax_hist = plt.subplot2grid((K + 1, 2), (i + 1, 0))
                ax_hist.hist(resids[:, i], bins=30,
                             density=True, alpha=0.7, color='skyblue')
                ax_hist.set_title(
                    f"Histogram of Residual {i} ({self.columns[i]})")
                ax_hist.set_xlabel("Residual Value")
                ax_hist.set_ylabel("Density")
                ax_hist.grid(True, alpha=0.3)
                ax_qq = plt.subplot2grid((K + 1, 2), (i + 1, 1))
                probplot(resids[:, i], dist="norm", plot=ax_qq)
                ax_qq.set_title(
                    f"Q–Q Plot for Residual {i} ({self.columns[i]})")
                ax_qq.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return Diagnosis

    def predict(self, n_periods=10, plot=True):
        if n_periods is None:
            n_periods = self.forecast_h
        A = np.array(self.best_model.get('A', np.zeros(
            (0, self.data.shape[1], self.data.shape[1]))))
        B = np.array(self.best_model.get('B', np.zeros(
            (0, self.data.shape[1], self.data.shape[1]))))
        Cst = np.ravel(self.best_model.get(
            'Cst', np.zeros((self.data.shape[1],))))
        p = self.best_model.get('p', A.shape[0]) if A.size else 0
        q = self.best_model.get('q', B.shape[0]) if B.size else 0
        K = self.data.shape[1]
        Y = self.data.to_numpy()
        resids = self.best_model.get('residuals', np.zeros((K, 0)))
        Sigma = np.cov(resids, bias=True) if resids.size else np.eye(K) * 1e-8
        psi = [np.eye(K)]
        for h in range(1, n_periods + 1):
            psi_h = np.zeros((K, K))
            for i in range(1, min(p + 1, h + 1)):
                psi_h += A[i - 1] @ psi[h - i]
            if h <= q:
                psi_h += B[h - 1]
            psi.append(psi_h)
        forecasts = np.zeros((n_periods, K))
        lag_buffer = [Y[-i - 1].copy() for i in range(p)] if p > 0 else []
        resid_buffer = [resids[:, -i - 1].copy()
                        for i in range(q)] if q > 0 else []
        for h in range(n_periods):
            y_hat = Cst.copy()
            for i in range(len(lag_buffer)):
                y_hat += A[i] @ lag_buffer[i]
            for j in range(len(resid_buffer)):
                y_hat += B[j] @ resid_buffer[j]
            forecasts[h, :] = y_hat
            if p > 0:
                lag_buffer = [y_hat] + lag_buffer[:-1]
            if q > 0:
                resid_buffer = [np.zeros(K)] + resid_buffer[:-1]
        forecast_vars = np.zeros((n_periods, K))
        for h in range(1, n_periods + 1):
            V = np.zeros((K, K))
            for s in range(h):
                V += psi[s] @ Sigma @ psi[s].T
            forecast_vars[h - 1, :] = np.maximum(np.diag(V), 0)
        se = np.sqrt(forecast_vars)
        z = norm.ppf(1 - self.ci_alpha / 2)
        ci_lower = forecasts - z * se
        ci_upper = forecasts + z * se
        if isinstance(self.data.index, pd.DatetimeIndex):
            freq = self.data.index.freq or pd.infer_freq(
                self.data.index) or 'D'
            start = self.data.index[-1]
            forecast_index = pd.date_range(
                start=start + (self.data.index.freq or pd.Timedelta(days=1)), periods=n_periods, freq=freq)
        else:
            forecast_index = range(len(self.data), len(self.data) + n_periods)
        forecast_df = pd.DataFrame(
            forecasts, index=forecast_index, columns=self.columns)
        ci_lower_df = pd.DataFrame(
            ci_lower, index=forecast_index, columns=self.columns)
        ci_upper_df = pd.DataFrame(
            ci_upper, index=forecast_index, columns=self.columns)
        if plot:
            T = len(self.data)
            whole = pd.concat([self.data, forecast_df], axis=0)
            fig, axes = plt.subplots(len(self.columns), 1, figsize=(
                12, 4 * len(self.columns)), sharex=True)
            if len(self.columns) == 1:
                axes = [axes]
            for i, col in enumerate(self.columns):
                ax = axes[i]
                time_range = np.arange(T + n_periods)
                ax.plot(time_range[:T], whole[col].iloc[:T],
                        'b-', label='Historical', linewidth=1.5)
                ax.plot(time_range[T - 1:T + n_periods], whole[col].iloc[T - 1:T + n_periods],
                        'k--', linewidth=1.5, label='Forecast' if i == 0 else "")
                ax.fill_between(time_range[T:], ci_lower_df[col], ci_upper_df[col],
                                color='skyblue', alpha=0.4,
                                label='95% CI' if i == 0 else "")
                ax.axvline(T - 1, color='gray', linestyle=':', linewidth=1)
                ax.set_title(f'Forecast for {col}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
            plt.tight_layout()
            plt.show()

        return {'point': forecast_df, 'ci_lower': ci_lower_df, 'ci_upper': ci_upper_df}

    def simulate(self, n_periods=100, plot=True):
        if self.best_model is None:
            raise ValueError(
                "No fitted model available. Fit model before simulate().")
        A = np.array(self.best_model.get('A', np.zeros(
            (0, self.data.shape[1], self.data.shape[1]))))
        B = np.array(self.best_model.get('B', np.zeros(
            (0, self.data.shape[1], self.data.shape[1]))))
        Cst = np.ravel(self.best_model.get(
            'Cst', np.zeros((self.data.shape[1],))))
        p = self.best_model.get('p', A.shape[0]) if A.size else 0
        q = self.best_model.get('q', B.shape[0]) if B.size else 0
        K = self.data.shape[1]
        Y_hist = [self.data.to_numpy()[-i - 1].copy()
                  for i in range(p)] if p > 0 else []
        resids = self.best_model.get('residuals', np.zeros((K, 0)))
        T_res = resids.shape[1] if resids is not None else 0
        if q > 0 and T_res > 0:
            E_hist = [resids[:, -i - 1].copy() for i in range(q)]
        else:
            E_hist = [np.zeros(K) for _ in range(q)]
        rng = np.random.default_rng()
        use_boot = T_res > 0
        Sigma = np.cov(resids, bias=True) if T_res > 0 else np.eye(K) * 1e-8
        Y_sim = np.zeros((n_periods, K))
        for t in range(n_periods):
            if use_boot:
                idx = rng.integers(0, T_res)
                eps_t = resids[:, idx].copy()
            else:
                eps_t = rng.multivariate_normal(np.zeros(K), Sigma)
            y_pred = Cst.copy()
            for i in range(p):
                y_pred += A[i] @ (Y_hist[i] if i <
                                  len(Y_hist) else np.zeros(K))
            for j in range(q):
                y_pred += B[j] @ (E_hist[j] if j <
                                  len(E_hist) else np.zeros(K))
            y_t = y_pred + eps_t
            Y_sim[t, :] = y_t
            if p > 0:
                Y_hist = [y_t] + \
                    Y_hist[:-1] if len(Y_hist) >= p else [y_t] + Y_hist
            if q > 0:
                E_hist = [eps_t] + \
                    E_hist[:-1] if len(E_hist) >= q else [eps_t] + E_hist
        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for i in range(K):
                axes[i].plot(
                    Y_sim[:, i], label=f'Simulated {self.data.columns[i]}')
                axes[i].set_title(
                    f'Simulated Series for {self.data.columns[i]}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True)
            plt.tight_layout()
            plt.show()
        return {'simulations': Y_sim}

    def impulse_res(self, h=10, bootstrap=True, n_boot=1000, plot=True):
        if self.best_model is None:
            raise ValueError(
                "No fitted model available. Fit model before impulse_res().")
        A = np.array(self.best_model.get('A', np.zeros(
            (0, self.data.shape[1], self.data.shape[1]))))
        B = np.array(self.best_model.get('B', np.zeros(
            (0, self.data.shape[1], self.data.shape[1]))))
        resids = self.best_model.get(
            'residuals', np.zeros((self.data.shape[1], 0)))
        K = self.data.shape[1]
        p = self.best_model.get('p', A.shape[0]) if A.size else 0
        q = self.best_model.get('q', B.shape[0]) if B.size else 0

        def compute_psi(A, B, H):
            psi = [np.eye(K)]
            for h in range(1, H + 1):
                psi_h = np.zeros((K, K))
                for i in range(1, min(p, h) + 1):
                    psi_h += A[i - 1] @ psi[h - i]
                if 1 <= h <= q:
                    psi_h += B[h - 1]
                psi.append(psi_h)
            return psi
        psi = compute_psi(A, B, h)
        Sigma = np.cov(resids, bias=True) if resids.size else np.eye(K) * 1e-8
        P = None
        if isinstance(self.best_model.get('Ph0'), np.ndarray):
            try:
                P = np.linalg.inv(self.best_model['Ph0'])
            except Exception:
                P = None
        if P is None:
            try:
                P = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(Sigma)
                vals_clipped = np.clip(vals, 0.0, None)
                P = vecs @ np.diag(np.sqrt(vals_clipped))
        irf = np.zeros((h + 1, K, K))
        for i in range(h + 1):
            irf[i, :, :] = psi[i] @ P

        result = {'irf': irf}
        if bootstrap and resids.size and n_boot > 0:
            all_boot = np.zeros((n_boot, h + 1, K, K))
            rng = np.random.default_rng()
            T_res = resids.shape[1]
            for b in range(n_boot):

                idx = rng.integers(0, T_res, size=T_res)
                boot_res = resids[:, idx]
                Sigma_b = np.cov(boot_res, bias=True)
                try:
                    P_b = np.linalg.cholesky(Sigma_b)
                except np.linalg.LinAlgError:
                    vals, vecs = np.linalg.eigh(Sigma_b)
                    vals_clipped = np.clip(vals, 0.0, None)
                    P_b = vecs @ np.diag(np.sqrt(vals_clipped))
                for i in range(h + 1):
                    all_boot[b, i, :, :] = psi[i] @ P_b
            ci_lower = np.percentile(all_boot, 2.5, axis=0)
            ci_upper = np.percentile(all_boot, 97.5, axis=0)
            result.update({'ci_lower': ci_lower, 'ci_upper': ci_upper})
        if plot:
            fig, axes = plt.subplots(K, K, figsize=(15, 15), sharex=True)
            axes = axes.flatten() if K > 1 else [axes]
            for i in range(K):
                for j in range(K):
                    idx = i * K + j
                    axes[idx].plot(range(h + 1), irf[:, i, j],
                                   label=f'Shock {self.data.columns[j]} → {self.data.columns[i]}')
                    if 'ci_lower' in result and 'ci_upper' in result:
                        axes[idx].fill_between(range(h + 1), result['ci_lower'][:, i, j],
                                               result['ci_upper'][:, i, j], alpha=0.3, color='red', label='95% CI')
                    axes[idx].set_title(
                        f'{self.data.columns[i]} response to {self.data.columns[j]} shock')
                    axes[idx].set_xlabel('Horizon')
                    axes[idx].set_ylabel('Response')
                    axes[idx].grid(True)
                    axes[idx].legend()
            plt.tight_layout()
            plt.show()

        return result

    def FEVD(self, h=10, plot=True):
        if self.best_model is None:
            raise ValueError(
                "No fitted model available. Fit model before FEVD().")
        K = self.data.shape[1]
        irf_result = self.impulse_res(h=h, bootstrap=False, plot=False)
        irf = irf_result['irf']
        fevd = np.zeros((h, K, K))
        mse = np.zeros((h, K))
        for i in range(h):
            horizon = i + 1
            for j in range(K):
                mse[i, j] = np.sum(irf[:horizon, j, :] ** 2)
                for k in range(K):
                    contribution = np.sum(irf[:horizon, j, k] ** 2)
                    fevd[i, j, k] = contribution / \
                        mse[i, j] if mse[i, j] != 0 else 0
            for j in range(K):
                total = np.sum(fevd[i, j, :])
                if total > 0:
                    fevd[i, j, :] /= total
        if plot:
            fig, axes = plt.subplots(K, 1, figsize=(10, 4 * K), sharex=True)
            axes = [axes] if K == 1 else axes
            for j in range(K):
                bottom = np.zeros(h)
                for k in range(K):
                    axes[j].bar(range(1, h + 1), fevd[:, j, k], bottom=bottom,
                                label=f'Shock from {self.data.columns[k]}')
                    bottom += fevd[:, j, k]
                axes[j].set_title(f'FEVD for {self.data.columns[j]}')
                axes[j].set_xlabel('Horizon')
                axes[j].set_ylabel('Variance Contribution')
                axes[j].legend()
                axes[j].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return fevd
