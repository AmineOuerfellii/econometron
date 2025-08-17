import pandas as pd
from econometron.utils.estimation.Regression import ols_estimator
import numpy as np
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
class Localprojirf:
    def __init__(self,
                 data: pd.DataFrame,
                 endogenous_vars: List[str],
                 exogenous_vars: Optional[List[str]] = None,
                 max_horizon: int = 20,
                 lags: int = 4,
                 constant: bool = True,
                 date_col: Optional[str] = None):
        self.data = data.copy()
        self.endogenous_vars = list(endogenous_vars)
        self.exogenous_vars = [] if exogenous_vars is None else list(exogenous_vars)
        self.H = int(max_horizon)
        self.p = int(lags)
        self.constant = bool(constant)
        self.date_col = date_col
        if self.date_col is not None and self.date_col in self.data.columns:
            self.data = self.data.sort_values(self.date_col).reset_index(drop=True)
            self.data = self.data.set_index(self.date_col)
        self._prepared = None
        self.results_: Dict[str, Any] = {}
        self._shock_var = self.endogenous_vars[0]

    def set_shock(self, shock_var: str):
        if shock_var not in self.endogenous_vars and shock_var not in self.data.columns:
            raise ValueError("shock_var must be in endogenous_vars or data columns")
        self._shock_var = shock_var
        return self

    @staticmethod
    def _make_lags(s: pd.Series, p: int, name: str) -> pd.DataFrame:
        return pd.concat({f"{name}_L{i}": s.shift(i) for i in range(1, p+1)}, axis=1) if p>0 else pd.DataFrame(index=s.index)

    def _prepare(self) -> pd.DataFrame:
        parts = []
        for v in self.endogenous_vars:
            parts.append(self.data[v].astype(float).rename(v))
        for x in self.exogenous_vars:
            parts.append(self.data[x].astype(float).rename(x))
        base = pd.concat(parts, axis=1)
        lag_blocks = []
        for v in self.endogenous_vars:
            lag_blocks.append(self._make_lags(base[v], self.p, v))
        for x in self.exogenous_vars:
            lag_blocks.append(self._make_lags(base[x], self.p, x))
        Xlags = pd.concat(lag_blocks, axis=1) if len(lag_blocks)>0 else pd.DataFrame(index=base.index)
        out = pd.concat([base, Xlags], axis=1)
        self._prepared = out
        return out

    @staticmethod
    def _lead(series: pd.Series, h: int) -> pd.Series:
        return series.shift(-h)

    @staticmethod
    def _auto_hac_lags(T: int) -> int:
        return int(np.floor(4*(T/100)**(2/9)))

    @staticmethod
    def _nw_cov(X: np.ndarray, u: np.ndarray, L: int) -> np.ndarray:
        X = np.asarray(X, float)
        u = np.asarray(u, float).reshape(-1)
        T, k = X.shape
        XT = X.T
        S = (XT @ (u[:,None]*u[:,None] * X)) / T
        for l in range(1, L+1):
            w = 1 - l/(L+1)
            X0 = X[:-l]
            Xl = X[l:]
            u0 = u[:-l]
            ul = u[l:]
            Sl = (X0.T @ (u0[:,None] * (ul[:,None] * Xl))) / T
            S += w*(Sl + Sl.T)
        bread = np.linalg.pinv((XT @ X) / T)
        V = bread @ S @ bread / T
        return V

    def fit(self,
            y_vars: Optional[List[str]] = None,
            shock_var: Optional[str] = None,
            horizons: Optional[int] = None,
            overlap: bool = True,
            difference: bool = False,
            cumulate: bool = False,
            hac: bool = True,
            hac_lags: Optional[int] = None,
            lag_select: Optional[str] = None,
            lag_select_max: int = 8,
            lag_ic: str = 'AIC'):
        if horizons is None:
            H = self.H
        else:
            H = int(horizons)
        if shock_var is not None:
            self.set_shock(shock_var)
        if y_vars is None:
            y_list = [v for v in self.endogenous_vars if v != self._shock_var]
        else:
            for yv in y_vars:
                if yv not in self.data.columns:
                    raise ValueError(f"Unknown y_var {yv}")
            y_list = list(y_vars)
        base = self._prepare()
        if lag_select is not None:
            best_p, best_ic = 0, np.inf
            for p in range(0, int(lag_select_max)+1):
                self.p = p
                tmp = self._prepare()
                dep = self._lead(tmp[y_list[0]], 1)
                if difference:
                    dep = dep - tmp[y_list[0]]
                dep = dep.dropna()
                X = tmp[[self._shock_var]].copy()
                regX = []
                for v in self.endogenous_vars + self.exogenous_vars:
                    regX.append(self._make_lags(tmp[v], p, v))
                if len(regX)>0:
                    X = pd.concat([X] + regX, axis=1)
                X = X.loc[dep.index].dropna()
                dep = dep.loc[X.index].values.reshape(-1,1)
                beta, fitted, resid, res = ols_estimator(X.values, dep, add_intercept=self.constant)
                s2 = (res['resid']**2).mean()
                T = len(dep)
                k = X.shape[1] + (1 if self.constant else 0)
                icv = np.log(s2) + (2 if lag_ic.upper()=='AIC' else np.log(T))*k/T
                if icv < best_ic:
                    best_ic, best_p = icv, p
            self.p = best_p
            base = self._prepare()
        results = {}
        for yv in y_list:
            rows = []
            full_betas: List[np.ndarray] = []
            full_covs: List[np.ndarray] = []
            for h in range(0, H+1):
                if cumulate:
                    dep = base[yv].shift(-h) - base[yv].shift(-1)
                else:
                    dep = base[yv].shift(-h)
                if difference:
                    dep = dep - base[yv]
                dep = dep.dropna()
                if not overlap and h>0:
                    dep = dep.iloc[::(h+1)]
                X = base[[self._shock_var]].copy()
                regX = []
                for v in self.endogenous_vars + self.exogenous_vars:
                    regX.append(self._make_lags(base[v], self.p, v))
                if len(regX)>0:
                    X = pd.concat([X] + regX, axis=1)
                X = X.loc[dep.index].dropna()
                dep = dep.loc[X.index].values.reshape(-1,1)
                beta, fitted, resid, res = ols_estimator(X.values, dep, add_intercept=self.constant)
                Tn = X.shape[0]
                k = X.shape[1] + (1 if self.constant else 0)
                if hac:
                    L = self._auto_hac_lags(Tn) if hac_lags is None else int(hac_lags)
                else:
                    L = 0
                Xcov = np.column_stack([np.ones(Tn), X.values]) if self.constant else X.values
                if L>0:
                    V = self._nw_cov(Xcov, res['resid'].flatten(), L)
                else:
                    XtX_inv = np.linalg.pinv(Xcov.T @ Xcov)
                    sigma2 = float((res['resid']**2).mean())
                    V = XtX_inv * sigma2
                idx = 1 if self.constant else 0
                bshock = beta[idx].item()
                se = np.sqrt(max(V[idx, idx].item(), 0))
                z = bshock / se if se>0 else np.nan
                row = {"h": h, "beta": bshock, "se": se, "z": z, "N": int(Tn), "hac_L": int(L)}
                rows.append(row)
                full_betas.append(beta.reshape(-1))
                full_covs.append(V)
            tbl = pd.DataFrame(rows).set_index("h")
            results[yv] = {"table": tbl, "betas_full": full_betas, "covs_full": full_covs,
                           "settings": {"p": self.p, "H": H, "shock": self._shock_var,
                                         "overlap": overlap, "difference": difference,
                                         "cumulate": cumulate, "constant": self.constant}}
        self.results_ = {"by_y": results, "meta": {"endogenous": self.endogenous_vars,
                                                     "exogenous": self.exogenous_vars}}
        return self

    def get_irf(self, y_var: str, level: float = 0.90) -> pd.DataFrame:
        if not self.results_:
            raise RuntimeError("Call fit() first.")
        if y_var not in self.results_["by_y"]:
            raise ValueError(f"{y_var} not in results. Available: {list(self.results_['by_y'].keys())}")
        tab = self.results_["by_y"][y_var]["table"].copy()
        from scipy.stats import norm
        q = norm.ppf(0.5 + level/2)
        tab["lo"] = tab["beta"] - q*tab["se"]
        tab["hi"] = tab["beta"] + q*tab["se"]
        return tab

    def long_run(self, y_var: str, H_star: Optional[int] = None, level: float = 0.90) -> Dict[str, float]:
        tab = self.get_irf(y_var, level)
        if H_star is None:
            H_star = int(tab.index.max())
        sub = tab.loc[:H_star]
        bsum = float(sub["beta"].sum())
        vsum = float((sub["se"]**2).sum())
        se = np.sqrt(vsum)
        from scipy.stats import norm
        q = norm.ppf(0.5 + level/2)
        return {"H_star": H_star, "lr": bsum, "se": float(se), "lo": float(bsum - q*se), "hi": float(bsum + q*se)}

    def summary(self) -> Dict[str, Any]:
        return self.results_

    def plot_irf(self, y_var: str, title: Optional[str] = None):
        """Plot IRF for a specific response variable with ±1.96 SE bands."""
        if not self.results_:
            raise RuntimeError("Must call fit() first")
        if y_var not in self.results_["by_y"]:
            raise ValueError(f"{y_var} not in results. Available: {list(self.results_['by_y'].keys())}")
        
        import matplotlib.pyplot as plt
        
        irf = self.results_["by_y"][y_var]["table"]
        ci_upper = irf["beta"] + 1.96 * irf["se"]
        ci_lower = irf["beta"] - 1.96 * irf["se"]

        plt.figure(figsize=(7, 5))
        plt.plot(irf.index, irf["beta"], label="IRF", marker="o")
        plt.fill_between(irf.index, ci_lower, ci_upper, color="gray", alpha=0.3, label="95% CI")
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Horizon")
        plt.ylabel(f"Response of {y_var}")
        plt.title(title or f"Local Projection IRF for {y_var} to {self.results_['by_y'][y_var]['settings']['shock']}")
        plt.legend()
        plt.tight_layout()
        plt.show()