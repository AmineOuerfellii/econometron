import pandas as pd
from econometron.utils.estimation.Regression import ols_estimator
import numpy as np
from typing import List, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist


class Localprojirf:
    def __init__(self,data: pd.DataFrame,endogenous_vars: List[str],exogenous_vars: Optional[List[str]] = None,max_horizon: int = 8,lags: Union[int, List[int]] = [1, 2],constant: bool = True,
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
        """
        for var in impulse_vars:
            if var not in self.endogenous_vars and var not in self.exogenous_vars:
                raise ValueError(
                    f"Impulse variable {var} must be in endogenous or exogenous variables")
        self._impulse_vars = list(impulse_vars)
        return self

    @staticmethod
    def _make_lags(s: pd.Series, lags: List[int], name: str) -> pd.DataFrame:
        """Create lagged variables for specific lag orders."""
        if not lags:
            return pd.DataFrame(index=s.index)
        lag_dict = {}
        for lag in lags:
            lag_dict[f"{name}_L{lag}"] = s.shift(lag)
        return pd.concat(lag_dict, axis=1)

    def _prepare(self) -> pd.DataFrame:
        """Prepare the dataset with lags, matching Stata's data preparation."""
        parts = []
        for v in self.endogenous_vars:
            parts.append(self.data[v].astype(float).rename(v))
        for x in self.exogenous_vars:
            parts.append(self.data[x].astype(float).rename(x))

        base = pd.concat(parts, axis=1)

        # Create all lags
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
        """Automatic lag selection for HAC standard errors (Newey-West)."""
        return int(np.floor(4 * (T / 100) ** (2/9)))

    @staticmethod
    def _nw_cov(X: np.ndarray, u: np.ndarray, L: int) -> np.ndarray:
        """Newey-West HAC covariance matrix estimator."""
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

    def _build_regression_data(self, response_var: str, impulse_var: str, horizon: int,
                               base: pd.DataFrame) -> tuple:
        """
        Build regression data for a specific response-impulse-horizon combination.

        This method correctly handles the timing and variable selection to match Stata.
        """
        if impulse_var in self.exogenous_vars:
            # For exogenous variables, estimate dynamic multipliers
            # y_{t+h} = φ_{ikh} x_{k,t} + Σ controls_{t} + u_{t+h}

            # Dependent variable at time t+h
            dep = base[response_var].shift(-horizon)

            # Build regressors (all dated at time t)
            X_parts = []

            # Impulse variable (contemporaneous, at time t)
            X_parts.append(base[impulse_var].to_frame(impulse_var))
            impulse_col_name = impulse_var

            # Controls: lags of all endogenous variables (dated t-1, t-2, etc.)
            for v in self.endogenous_vars:
                lag_block = self._make_lags(base[v], self.lags, v)
                if not lag_block.empty:
                    X_parts.append(lag_block)

            # Controls: lags of other exogenous variables (not the impulse variable itself)
            for x in self.exogenous_vars:
                if x != impulse_var:
                    lag_block = self._make_lags(base[x], self.lags, x)
                    if not lag_block.empty:
                        X_parts.append(lag_block)

            # Combine all regressors
            X = pd.concat(X_parts, axis=1)

        else:
            # For endogenous variables, estimate IRFs
            # y_{i,t+h-1} = θ_{ijh} y_{j,t-1} + Σ controls + u_{t+h-1}

            # Dependent variable
            if horizon == 1:
                dep = base[response_var].copy()  # y_{i,t}
            else:
                dep = base[response_var].shift(-(horizon-1))  # y_{i,t+h-1}

            # Build regressors
            X_parts = []

            # First lag of impulse variable (this is the IRF coefficient we want)
            X_parts.append(base[impulse_var].shift(
                1).to_frame(f"{impulse_var}_L1"))
            impulse_col_name = f"{impulse_var}_L1"

            # Control lags for endogenous variables
            for v in self.endogenous_vars:
                if v == impulse_var:
                    # For impulse variable, include higher-order lags as controls
                    control_lags = [l for l in self.lags if l > 1]
                else:
                    # For other endogenous variables, include all specified lags
                    control_lags = self.lags

                if control_lags:
                    lag_block = self._make_lags(base[v], control_lags, v)
                    if not lag_block.empty:
                        X_parts.append(lag_block)

            # Add lags of exogenous variables as controls
            for x in self.exogenous_vars:
                lag_block = self._make_lags(base[x], self.lags, x)
                if not lag_block.empty:
                    X_parts.append(lag_block)

            # Combine all regressors
            if X_parts:
                X = pd.concat(X_parts, axis=1)
            else:
                raise ValueError(
                    f"No valid regressors for {response_var} -> {impulse_var} at horizon {horizon}")

        return dep, X, impulse_col_name

    def fit(self,response_vars: Optional[List[str]] = None,impulse_vars: Optional[List[str]] = None,horizons: Optional[int] = None,
            difference: bool = False,cumulate: bool = False,hac: bool = False,hac_lags: Optional[int] = None,robust: bool = False,dfk: bool = False,
            small: bool = False):
        """
        Fit the local projection IRF model.

        Parameters:
        -----------
        response_vars : Optional[List[str]]
            List of response variables (default: all endogenous variables)
        impulse_vars : Optional[List[str]]
            List of impulse variables (default: all endogenous + exogenous variables)
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
            self._impulse_vars = self.endogenous_vars + self.exogenous_vars

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
                if imp_var in self.exogenous_vars:
                    horizon_range = range(0, H + 1)
                else:
                    horizon_range = range(1, H + 1)

                for h in horizon_range:
                    try:
                        dep, X, impulse_col_name = self._build_regression_data(
                            resp_var, imp_var, h, base)
                        if difference and h > 0:
                            dep = dep - base[resp_var]

                        if cumulate and h > 0:

                            if imp_var in self.exogenous_vars:
                                dep = sum(base[resp_var].shift(-i)
                                          for i in range(h + 1))
                            else:
                                dep = sum(base[resp_var].shift(-(i-1))
                                          for i in range(1, h + 1))

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
                            if hac_lags is None:
                                L = self._auto_hac_lags(T)
                            else:
                                L = int(hac_lags)
                            V = self._nw_cov(X_full, residuals, L)
                        elif robust:
                            XTX_inv = np.linalg.pinv(X_full.T @ X_full)
                            meat = X_full.T @ np.diag(residuals**2) @ X_full
                            V = XTX_inv @ meat @ XTX_inv
                        else:
                            XTX_inv = np.linalg.pinv(X_full.T @ X_full)
                            sigma2 = (residuals**2).sum() / df
                            V = XTX_inv * sigma2
                        if impulse_col_name in X_clean.columns:
                            impulse_idx = list(X_clean.columns).index(
                                impulse_col_name)
                            if self.constant:
                                impulse_idx += 1  # Account for constant term

                            if impulse_idx < len(beta):
                                beta_impulse = beta[impulse_idx].item()
                                se_impulse = np.sqrt(
                                    max(V[impulse_idx, impulse_idx].item(), 0))
                            else:
                                beta_impulse = 0.0
                                se_impulse = 0.0
                        else:
                            beta_impulse = 0.0
                            se_impulse = 0.0
                        if se_impulse > 0:
                            test_stat = beta_impulse / se_impulse
                            if small:
                                p_value = 2 * \
                                    (1 - t_dist.cdf(abs(test_stat), df))
                            else:
                                p_value = 2 * (1 - norm.cdf(abs(test_stat)))
                        else:
                            test_stat = np.nan
                            p_value = np.nan
                        if imp_var in self.exogenous_vars and h == 0:
                            horizon_label = "--"
                        else:
                            horizon_label = h

                        row = {
                            "h": horizon_label,
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

                    except Exception as e:
                        continue

                if rows:
                    tbl = pd.DataFrame(rows)
                    if not tbl.empty:
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

        # Calculate confidence intervals
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
        """Return full results dictionary."""
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
        """
        if not self.results_:
            raise RuntimeError("Must call fit() first")

        irf_data = self.get_irf(response_var, impulse_var, level)

        fig, ax = plt.subplots(figsize=figsize)
        numeric_horizons = []
        betas = []
        ci_lowers = []
        ci_uppers = []
        for idx, row in irf_data.iterrows():
            h = row["h"]
            if h == "--":
                h_numeric = 0
            else:
                h_numeric = int(h)
            numeric_horizons.append(h_numeric)
            betas.append(row["beta"])
            ci_lowers.append(row["ci_lower"])
            ci_uppers.append(row["ci_upper"])

        ax.plot(numeric_horizons, betas, 'bo-',
                label="IRF", linewidth=2, markersize=4)
        ax.fill_between(numeric_horizons, ci_lowers, ci_uppers,
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
        """
        if not self.results_:
            raise RuntimeError("Must call fit() first")

        all_rows = []

        for resp_var in self.results_["by_response"]:
            for imp_var in self.results_["by_response"][resp_var]:
                table = self.results_[
                    "by_response"][resp_var][imp_var]["table"]

                for idx, row in table.iterrows():
                    h = row["h"]
                    if h == "--":
                        horizon_str = "--."
                    else:
                        horizon_str = f"F{int(h)}."
                    ci_lower = row["beta"] - 1.96 * row["se"]
                    ci_upper = row["beta"] + 1.96 * row["se"]

                    stata_row = {
                        "impulse": imp_var,
                        "response": resp_var,
                        "horizon": horizon_str,
                        "coefficient": row["beta"],
                        "std_err": row["se"],
                        "t_z": row.get("t", row.get("z", np.nan)),
                        "p_value": row.get("pvalue", np.nan),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                    all_rows.append(stata_row)

        df = pd.DataFrame(all_rows)

        def horizon_sort_key(h):
            h_str = str(h)
            if h_str == "--.":
                return -1
            else:
                try:
                    if h_str.startswith("F"):
                        num_str = h_str[1:].rstrip(".")
                        return int(float(num_str))
                    else:
                        return int(float(h_str))
                except (ValueError, TypeError):
                    return 999

        df["horizon_sort"] = df["horizon"].apply(horizon_sort_key)
        df = df.sort_values(by=["impulse", "response", "horizon_sort"]).drop(
            "horizon_sort", axis=1)
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
        if self.exogenous_vars:
            output_lines.append(
                "Note: IRF coefficients for exogenous variables are dynamic multipliers.")
        print("\n".join(output_lines))
