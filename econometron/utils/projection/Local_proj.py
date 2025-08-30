import pandas as pd
from econometron.utils.estimation.Regression import ols_estimator
import numpy as np
from typing import List, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


class Localprojirf:
    def __init__(self,
                 data: pd.DataFrame,
                 endogenous_vars: List[str],
                 exogenous_vars: Optional[List[str]] = None,
                 max_horizon: int = 8,
                 lags: Union[int, List[int]] = [1, 2],
                 constant: bool = True,
                 date_col: Optional[str] = None):
        """
        Initialize Local Projection IRF estimator to match Stata's lpirf.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        endogenous_vars : List[str]
            List of endogenous variables
        exogenous_vars : Optional[List[str]]
            List of exogenous variables (for dynamic multipliers)
        max_horizon : int
            Maximum forecast horizon (default 8, matching Stata)
        lags : Union[int, List[int]]
            Lags to include. If int, includes lags 1 through lags.
            If list, includes specific lags (default [1,2] matching Stata)
        constant : bool
            Whether to include constant term
        date_col : Optional[str]
            Date column for sorting
        """
        self.data = data.copy()
        self.endogenous_vars = list(endogenous_vars)
        self.exogenous_vars = [] if exogenous_vars is None else list(
            exogenous_vars)
        self.H = int(max_horizon)
        if isinstance(lags, int):
            self.lags = list(range(1, lags + 1))
        else:
            self.lags = list(lags)
        self.constant = bool(constant)
        self.date_col = date_col
        if self.date_col is not None and self.date_col in self.data.columns:
            self.data = self.data.sort_values(
                self.date_col).reset_index(drop=True)
        self._prepared = None
        self.results_ = {}
        self._impulse_vars = self.endogenous_vars.copy()

    def set_impulse_vars(self, impulse_vars: List[str]):
        """
        Set which variables to treat as impulse variables.

        Parameters:
        -----------
        impulse_vars : List[str]
            List of variable names to treat as impulses

        Returns:
        --------
        Localprojirf
            Self, for method chaining

        Raises:
        -------
        ValueError
            If any impulse variable is not in endogenous or exogenous variables
        """
        for var in impulse_vars:
            if var not in self.endogenous_vars and var not in self.exogenous_vars:
                raise ValueError(
                    f"Impulse variable {var} must be in endogenous or exogenous variables")
        self._impulse_vars = list(impulse_vars)
        return self

    @staticmethod
    def _make_lags(s: pd.Series, lags: List[int], name: str) -> pd.DataFrame:
        """
        Create lagged variables for specific lag orders.

        Parameters:
        -----------
        s : pd.Series
            Input series
        lags : List[int]
            List of lag orders
        name : str
            Base name for lagged variables

        Returns:
        --------
        pd.DataFrame
            DataFrame containing lagged variables
        """
        if not lags:
            return pd.DataFrame(index=s.index)
        lag_dict = {}
        for lag in lags:
            lag_dict[f"{name}_L{lag}"] = s.shift(lag)
        return pd.concat(lag_dict, axis=1)

    def _prepare(self) -> pd.DataFrame:
        """
        Prepare the dataset with lags, matching Stata's data preparation.

        Returns:
        --------
        pd.DataFrame
            DataFrame with base variables and their lags
        """
        parts = []
        for v in self.endogenous_vars:
            parts.append(self.data[v].astype(float).rename(v))
        for x in self.exogenous_vars:
            parts.append(self.data[x].astype(float).rename(x))
        base = pd.concat(parts, axis=1)
        lag_blocks = []
        for v in self.endogenous_vars:
            lag_blocks.append(self._make_lags(base[v], self.lags, v))
        for x in self.exogenous_vars:
            lag_blocks.append(self._make_lags(base[x], self.lags, x))
        if lag_blocks:
            Xlags = pd.concat(lag_blocks, axis=1)
            out = pd.concat([base, Xlags], axis=1)
        else:
            out = base
        self._prepared = out
        return out

    @staticmethod
    def _auto_hac_lags(T: int) -> int:
        """
        Automatic lag selection for HAC standard errors (Newey-West).

        Parameters:
        -----------
        T : int
            Sample size

        Returns:
        --------
        int
            Number of lags for HAC estimation
        """
        return int(np.floor(4 * (T / 100) ** (2/9)))

    @staticmethod
    def _nw_cov(X: np.ndarray, u: np.ndarray, L: int) -> np.ndarray:
        """
        Newey-West HAC covariance matrix estimator.

        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        u : np.ndarray
            Residuals array
        L : int
            Number of lags for HAC estimation

        Returns:
        --------
        np.ndarray
            HAC covariance matrix
        """
        X = np.asarray(X, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        T, k = X.shape
        XT = X.T
        S = (XT @ np.diag(u**2) @ X) / T
        for l in range(1, L + 1):
            w = 1 - l / (L + 1)
            X_lead = X[l:]
            X_lag = X[:-l]
            u_lead = u[l:]
            u_lag = u[:-l]
            gamma_l = (X_lag.T @ np.diag(u_lag * u_lead) @ X_lead) / T
            S += w * (gamma_l + gamma_l.T)
        XTX_inv = np.linalg.pinv(XT @ X / T)
        V = XTX_inv @ S @ XTX_inv / T
        return V

    def fit(self,
            response_vars: Optional[List[str]] = None,
            impulse_vars: Optional[List[str]] = None,
            horizons: Optional[int] = None,
            difference: bool = False,
            cumulate: bool = False,
            hac: bool = False,
            hac_lags: Optional[int] = None,
            robust: bool = False,
            dfk: bool = False,
            small: bool = False):
        """
        Fit the local projection IRF model.

        Parameters:
        -----------
        response_vars : Optional[List[str]]
            List of response variables (default: all endogenous variables)
        impulse_vars : Optional[List[str]]
            List of impulse variables (default: all endogenous variables)
        horizons : Optional[int]
            Maximum forecast horizon (default: self.H)
        difference : bool
            If True, use differenced dependent variable
        cumulate : bool
            If True, use cumulative response
        hac : bool
            If True, use Newey-West HAC standard errors
        hac_lags : Optional[int]
            Number of lags for HAC estimation (default: automatic selection)
        robust : bool
            If True, use White robust standard errors
        dfk : bool
            If True, apply degrees of freedom correction
        small : bool
            If True, use t-distribution for inference; otherwise, use normal

        Returns:
        --------
        Localprojirf
            Self, for method chaining
        """
        H = self.H if horizons is None else int(horizons)
        if impulse_vars is not None:
            self.set_impulse_vars(impulse_vars)
        else:
            self._impulse_vars = self.endogenous_vars.copy()
        if response_vars is None:
            response_vars = self.endogenous_vars.copy()
        base = self._prepare()
        results = {}
        for resp_var in response_vars:
            resp_results = {}
            for imp_var in self._impulse_vars:
                if imp_var not in self.endogenous_vars and imp_var not in self.exogenous_vars:
                    continue
                rows = []
                full_betas = []
                full_covs = []
                for h in range(1, H + 1):
                    if imp_var in self.exogenous_vars:
                        dep = base[resp_var].shift(-h)
                        X_impulse = base[[imp_var]].copy()
                        control_parts = []
                        for v in self.endogenous_vars:
                            control_parts.append(
                                self._make_lags(base[v], self.lags, v))
                        for x in self.exogenous_vars:
                            if x != imp_var:
                                control_parts.append(
                                    self._make_lags(base[x], self.lags, x))
                        if control_parts:
                            X_controls = pd.concat(
                                [x for x in control_parts if not x.empty], axis=1)
                            X = pd.concat([X_impulse, X_controls], axis=1)
                        else:
                            X = X_impulse
                    else:
                        if h == 1:
                            dep = base[resp_var].copy()
                        else:
                            dep = base[resp_var].shift(-(h-1))
                        if difference:
                            dep = dep - base[resp_var]
                        if cumulate:
                            dep = sum(base[resp_var].shift(-i)
                                      for i in range(h))
                        X_parts = []
                        X_parts.append(base[imp_var].shift(
                            1).to_frame(f"{imp_var}_L1"))
                        for v in self.endogenous_vars:
                            if v == imp_var:
                                control_lags = [l for l in self.lags if l > 1]
                            else:
                                control_lags = self.lags
                            if control_lags:
                                X_parts.append(self._make_lags(
                                    base[v], control_lags, v))
                        non_empty_parts = [x for x in X_parts if not x.empty]
                        if non_empty_parts:
                            X = pd.concat(non_empty_parts, axis=1)
                        else:
                            continue
                    dep_notna = dep.notna()
                    X_notna = X.notna().all(axis=1)
                    valid_idx = dep_notna & X_notna
                    if not valid_idx.any():
                        continue
                    dep_clean = dep[valid_idx]
                    X_clean = X[valid_idx]
                    if len(dep_clean) == 0:
                        continue
                    y_array = dep_clean.values.reshape(-1, 1)
                    X_array = X_clean.values
                    beta, fitted, resid, res = ols_estimator(
                        X_array, y_array, add_intercept=self.constant)
                    T = len(dep_clean)
                    k = X_array.shape[1] + (1 if self.constant else 0)
                    df = T - k if dfk else T
                    if self.constant:
                        X_full = np.column_stack([np.ones(T), X_array])
                    else:
                        X_full = X_array
                    residuals = res['resid'].flatten()
                    if hac:
                        L = self._auto_hac_lags(
                            T) if hac_lags is None else int(hac_lags)
                        V = self._nw_cov(X_full, residuals, L)
                    elif robust:
                        XTX_inv = np.linalg.pinv(X_full.T @ X_full)
                        meat = X_full.T @ np.diag(residuals**2) @ X_full
                        V = XTX_inv @ meat @ XTX_inv
                    else:
                        XTX_inv = np.linalg.pinv(X_full.T @ X_full)
                        sigma2 = (residuals**2).sum() / df
                        V = XTX_inv * sigma2
                    impulse_idx = 1 if self.constant else 0
                    if impulse_idx < len(beta):
                        beta_impulse = beta[impulse_idx].item()
                        se_impulse = np.sqrt(
                            max(V[impulse_idx, impulse_idx].item(), 0))
                    else:
                        beta_impulse = 0.0
                        se_impulse = 0.0
                    if se_impulse > 0:
                        test_stat = beta_impulse / se_impulse
                        if small:
                            p_value = 2 * (1 - t_dist.cdf(abs(test_stat), df))
                        else:
                            p_value = 2 * (1 - norm.cdf(abs(test_stat)))
                    else:
                        test_stat = np.nan
                        p_value = np.nan
                    row = {
                        "h": h,
                        "beta": beta_impulse,
                        "se": se_impulse,
                        "t" if small else "z": test_stat,
                        "pvalue": p_value,
                        "N": int(T),
                        "df": int(df) if dfk else None
                    }
                    if hac:
                        row["hac_L"] = int(L)
                    rows.append(row)
                    full_betas.append(beta.reshape(-1))
                    full_covs.append(V)
                if rows:
                    tbl = pd.DataFrame(rows).set_index("h")
                    resp_results[imp_var] = {
                        "table": tbl,
                        "betas_full": full_betas,
                        "covs_full": full_covs
                    }
            results[resp_var] = resp_results
        self.results_ = {
            "by_response": results,
            "meta": {
                "endogenous": self.endogenous_vars,
                "exogenous": self.exogenous_vars,
                "impulse_vars": self._impulse_vars,
                "response_vars": response_vars,
                "lags": self.lags,
                "H": H,
                "settings": {
                    "difference": difference,
                    "cumulate": cumulate,
                    "constant": self.constant,
                    "hac": hac,
                    "robust": robust,
                    "dfk": dfk,
                    "small": small
                }
            }
        }
        return self

    def get_irf(self, response_var: str, impulse_var: str, level: float = 0.95) -> pd.DataFrame:
        """
        Get IRF table with confidence intervals.

        Parameters:
        -----------
        response_var : str
            Name of the response variable
        impulse_var : str
            Name of the impulse variable
        level : float
            Confidence level for intervals (default: 0.95)

        Returns:
        --------
        pd.DataFrame
            DataFrame with IRF estimates, standard errors, test statistics, p-values, and confidence intervals

        Raises:
        -------
        RuntimeError
            If fit() has not been called
        ValueError
            If response or impulse variable not found in results
        """
        if not self.results_:
            raise RuntimeError("Call fit() first.")
        if response_var not in self.results_["by_response"]:
            raise ValueError(
                f"Response variable {response_var} not found in results.")
        if impulse_var not in self.results_["by_response"][response_var]:
            raise ValueError(
                f"Impulse variable {impulse_var} not found for response {response_var}.")
        tab = self.results_[
            "by_response"][response_var][impulse_var]["table"].copy()
        if self.results_["meta"]["settings"]["small"]:
            df = tab["df"].iloc[0] if "df" in tab.columns and not pd.isna(
                tab["df"].iloc[0]) else len(tab)
            q = t_dist.ppf(0.5 + level/2, df)
        else:
            q = norm.ppf(0.5 + level/2)
        tab["ci_lower"] = tab["beta"] - q * tab["se"]
        tab["ci_upper"] = tab["beta"] + q * tab["se"]
        return tab

    def summary(self) -> Dict[str, Any]:
        """
        Return full results dictionary.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing estimation results
        """
        return self.results_

    def plot_irf(self, response_var: str, impulse_var: str,
                 level: float = 0.95, title: Optional[str] = None,
                 figsize: tuple = (10, 6)):
        """
        Plot IRF with confidence bands.

        Parameters:
        -----------
        response_var : str
            Name of the response variable
        impulse_var : str
            Name of the impulse variable
        level : float
            Confidence level for bands (default: 0.95)
        title : Optional[str]
            Optional plot title
        figsize : tuple
            Figure size as (width, height)

        Returns:
        --------
        tuple
            Matplotlib figure and axes objects

        Raises:
        -------
        RuntimeError
            If fit() has not been called
        """
        if not self.results_:
            raise RuntimeError("Must call fit() first")
        irf_data = self.get_irf(response_var, impulse_var, level)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(irf_data.index, irf_data["beta"], 'bo-',
                label="IRF", linewidth=2, markersize=4)
        ax.fill_between(irf_data.index, irf_data["ci_lower"], irf_data["ci_upper"],
                        alpha=0.3, color='gray', label=f'{int(level*100)}% CI')
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_xlabel("Horizon")
        ax.set_ylabel(f"Response of {response_var}")
        ax.set_title(title or f"IRF: {response_var} to {impulse_var}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig, ax

    def get_summary(self) -> str:
        """
        Return a formatted string summary of IRF results, mimicking Stata output.

        Returns:
        --------
        str
            Formatted string with IRF results, grouped by impulse variable

        Raises:
        -------
        RuntimeError
            If fit() has not been called
        """
        if not self.results_:
            raise RuntimeError("Must call fit() first")
        all_rows = []
        for resp_var in self.results_["by_response"]:
            for imp_var in self.results_["by_response"][resp_var]:
                table = self.results_[
                    "by_response"][resp_var][imp_var]["table"]
                for h, row in table.iterrows():
                    stata_row = {
                        "impulse": imp_var,
                        "response": resp_var,
                        "horizon": f"F{h}.",
                        "coefficient": row["beta"],
                        "std_err": row["se"],
                        "t_z": row.get("t", row.get("z", np.nan)),
                        "p_value": row.get("pvalue", np.nan),
                        "ci_lower": row["beta"] - 1.96 * row["se"],
                        "ci_upper": row["beta"] + 1.96 * row["se"],
                    }
                    all_rows.append(stata_row)
        df = pd.DataFrame(all_rows)
        df = df.sort_values(by=["impulse", "response", "horizon"])
        output_lines = []
        for imp, g in df.groupby("impulse"):
            output_lines.append("=" * 60)
            output_lines.append(f"Impulse variable: {imp}")
            output_lines.append("=" * 60)
            header = f"{'Response':<15}{'Horizon':<8}{'Coef.':>12}{'Std.Err.':>12}{'t/z':>8}{'P>|t|':>10}{'[95% Conf. Int.]':>20}"
            output_lines.append(header)
            output_lines.append("-" * len(header))
            for _, r in g.iterrows():
                line = f"{r['response']:<15}{r['horizon']:<8}{r['coefficient']:>12.4f}{r['std_err']:>12.4f}{r['t_z']:>8.2f}{r['p_value']:>10.3f}{r['ci_lower']:>10.3f} {r['ci_upper']:<10.3f}"
                output_lines.append(line)
            output_lines.append("")
        return "\n".join(output_lines)
