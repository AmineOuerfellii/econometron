from typing import Union, Dict, List, Optional ,Callable
import pandas as pd
import numpy as np
from econometron.Models.dynamicsge import linear_dsge
from econometron.utils.estimation.MLE import simulated_annealing_kalman, genetic_algorithm_kalman
from econometron.utils.estimation.Bayesian import rwm_kalman, compute_proposal_sigma, make_prior_function
from econometron.filters import kalman_objective, Kalman, kalman_smooth
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
from econometron.utils.estimation.results import compute_stats
import logging
from scipy.optimize import minimize
import ast
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot, skew, kurtosis
from statsmodels.tsa.stattools import acf
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SS_Model:
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, pd.Series],parameters: dict, model: linear_dsge = None,name:str='STate Space Model', optimizer: str = 'L-BFGS-B', estimation_method: str = 'MLE', constraints: dict = None):
        """
        Initializes the State Space Model with the given parameters.
        Parameters:
        - data (Union[np.ndarray, pd.DataFrame, pd.Series]): The observed data.
        - parameters (dict): Model parameters.
        - P (np.ndarray): Estimate error covariance.
        - x0 (np.ndarray): Initial state estimate.
        - model (econometron.Models.dynamicsge.linear_dsge, optional): The linear DSGE model.
        - optimizer (str, optional): The optimization algorithm to use.
        - estimation_method (str, optional): The estimation method to use.
        - constraints (dict, optional): Constraints for the optimization.
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            self.data = data.values
        else:
            self.data = data
        self.name=name if name else 'state space model'
        self.parameters = parameters
        self.optimizer = optimizer
        self.technique = estimation_method
        self.constraints = constraints
        self.A = None
        self.Q = None
        self.D = None
        self.R = None
        self.model = model
        self._predef={}
        self.fixed_params={}
        self._derived_params={}
        self.validated_entries_=False
        self._predef_expressions={}
        self.state_updater=None
        ####
        self.result=None
    def validate_entries_(self):
        if self.data.shape[0] > self.data.shape[1]:
            raise ValueError("Data Must be in The shape of (N_vars ,Time period)")
        if self.model is None and (self.A is None or self.D is None or self.Q is None or self.R is None):
            raise ValueError(
                "Model OR transition matrix A, observation matrix D, Q, and R must be defined.")
        if self.model is not None and (self.A is not None or self.D is not None):
            raise ValueError(
                "Model cannot be defined alongside transition matrix A or observation matrix C.")
        optimizers_list = ['sa', 'ga', 'trust_constr', 'bfgs', 'powell']
        techniques_list = ['MLE', 'Bayesian']
        if self.technique not in techniques_list:
            raise ValueError(
                f"Estimation method must be one of {techniques_list}.")
        if self.technique == 'Bayesian':
            if self.optimizer is not None:
                self.optimizer = None
        if self.technique == 'MLE' and self.optimizer is None:
            self.optimizer = 'sa'
        if self.optimizer.lower() not in optimizers_list:
            raise ValueError(f"Optimizer must be one of {optimizers_list}.")
        if self.constraints and self.optimizer != 'trust_constr':
            logger.info(f"The constraints aren't going to be taken into account since you've set your optimizer as {self.optimizer}")
        self.validated_entries_=True    
    def set_transition_mat(self, A: Union[np.ndarray, np.matrix, Callable], params: Optional[Dict] = None):
        """Set the state transition matrix A, optionally using parameters."""
        if isinstance(A, Callable):
            # Use provided params or self.parameters for validation
            test_params = params if params is not None else self.parameters
            try:
                test_A = A(test_params)
                A_array = np.asarray(test_A)
            except Exception as e:
                raise ValueError(f"Callable for A failed with parameters: {str(e)}")
        else:
            A_array = np.asarray(A)

        if A_array.shape[0] != A_array.shape[1]:
            raise ValueError("Transition matrix A must be square.")
        if not np.all(np.isfinite(A_array)):
            raise ValueError("Transition matrix A contains non-numeric or infinite values.")
        self.A = A
        return self.A

    def set_design_mat(self, C: Union[np.ndarray, np.matrix, Callable], params: Optional[Dict] = None):
        """Set the observation matrix C, optionally using parameters."""
        if self.A is None and self.model is None:
            raise ValueError("Set transition matrix or model before setting observation matrix.")
        A_shape = self.model.n_states if self.model else (self.A(self.parameters) if isinstance(self.A, Callable) else self.A).shape[0]
        if isinstance(C, Callable):
            test_params = params if params is not None else self.parameters
            try:
                test_C = C(test_params)
                C_array = np.asarray(test_C)
            except Exception as e:
                raise ValueError(f"Callable for C failed with parameters: {str(e)}")
        else:
            C_array = np.asarray(C)

        if C_array.shape[1] != A_shape or C_array.shape[0] != self.data.shape[1]:
            raise ValueError("Observation matrix C dimensions are not compatible with transition matrix or data.")
        if not np.all(np.isfinite(C_array)):
            raise ValueError("Observation matrix C contains non-numeric or infinite values.")
        self.C = C
        return self.C

    def set_state_cov(self, Q: Union[np.ndarray, np.matrix, Callable], params: Optional[Dict] = None):
        """Set the state noise covariance matrix Q, optionally using parameters."""
        if self.model is None and self.A is None:
            raise ValueError("Set transition matrix or model before setting state covariance matrix.")
        A_shape = self.model.n_states if self.model else (self.A(self.parameters) if isinstance(self.A, Callable) else self.A).shape[0]
        if isinstance(Q, Callable):
            test_params = params if params is not None else self.parameters
            try:
                test_Q = Q(test_params)
                Q_array = np.asarray(test_Q)
            except Exception as e:
                raise ValueError(f"Callable for Q failed with parameters: {str(e)}")
        else:
            Q_array = np.asarray(Q)

        if Q_array.shape[0] != Q_array.shape[1] or Q_array.shape[0] != A_shape:
            raise ValueError("State covariance matrix Q must be square and match transition matrix dimensions.")
        if not np.all(np.linalg.eigvals(Q_array) >= 0):
            raise ValueError("State covariance matrix Q must be positive semi-definite.")
        if not np.all(np.isfinite(Q_array)):
            raise ValueError("State covariance matrix Q contains non-numeric or infinite values.")
        self.Q = Q
        return self.Q

    def set_obs_cov(self, R: Union[np.ndarray, np.matrix, Callable], params: Optional[Dict] = None):
        """Set the observation noise covariance matrix R, optionally using parameters."""
        logger.info("R needs to be set as a full covariance matrix")
        ex_dim_R = self.model.n_controls if self.model else (self.D(self.parameters) if isinstance(self.D, Callable) else self.D).shape[0]
        if isinstance(R, Callable):
            test_params = params if params is not None else self.parameters
            try:
                test_R = R(test_params)
                R_array = np.asarray(test_R)
            except Exception as e:
                raise ValueError(f"Callable for R failed with parameters: {str(e)}")
        else:
            R_array = np.asarray(R)

        if R_array.shape[0] != R_array.shape[1] or R_array.shape[0] != ex_dim_R:
            raise ValueError(f"Observation covariance matrix R must be square and match expected dimension {ex_dim_R}.")
        if not np.all(np.linalg.eigvals(R_array) >= 0):
            logger.warning("Observation covariance matrix R must be positive semi-definite.")
        if not np.all(np.isfinite(R_array)):
            raise ValueError("Observation covariance matrix R contains non-numeric or infinite values.")
        self.R = R
        return self.R

    def set_Design_mat(self, D: Union[np.ndarray, np.matrix, Callable], params: Optional[Dict] = None):
        """Set the control matrix D, optionally using parameters."""
        if self.A is None and self.model is None:
            raise ValueError("Set transition matrix or model before setting control matrix.")
        ex_dim_d_r = self.model.n_controls if self.model else self.data.shape[0]
        ex_dim_d_c = self.model.n_states if self.model else (self.A(self.parameters) if isinstance(self.A, Callable) else self.A).shape[0]
        if isinstance(D, Callable):
            test_params = params if params is not None else self.parameters
            try:
                test_D = D(test_params)
                D_array = np.asarray(test_D)
            except Exception as e:
                raise ValueError(f"Callable for D failed with parameters: {str(e)}")
        else:
            D_array = np.asarray(D)

        if D_array.shape[0] != ex_dim_d_r or D_array.shape[1] != ex_dim_d_c:
            raise ValueError(f"Control matrix D must have shape ({ex_dim_d_r}, {ex_dim_d_c}).")
        if not np.all(np.isfinite(D_array)):
            raise ValueError("Control matrix D contains non-numeric or infinite values.")
        self.D = D
        return self.D

    def _make_state_space_updater(self, base_params: dict):
        def update_state_space(params):
            full_params = base_params.copy()
            full_params.update(params)
            if hasattr(self, '_predef_expressions') and self._predef_expressions:
                self.define_parameter(self._predef_expressions)
                full_params.update(self._derived_params) 
            if self.model is not None:
                D, A = self.model.solve_RE_model(full_params)
            else:
                if self.A is None or self.D is None:
                    raise ValueError("Transition matrix A and observation matrix D must be set if no solver is provided.")
                A = self.A(full_params) if isinstance(self.A, Callable) else self.A
                D = self.D(full_params) if isinstance(self.D, Callable) else (self.D if self.D is not None else np.zeros((self.x0.shape[0], 1)))

            R = self.R(full_params) if isinstance(self.R, Callable) else self.R
            Q = self.Q(full_params) if isinstance(self.Q, Callable) else self.Q

            if not np.all(np.isfinite(A)) or not np.all(np.isfinite(D)) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(Q)):
                raise ValueError("Computed A, D, R, or Q contains non-numeric or infinite values.")
            if not np.all(np.linalg.eigvals(R @ R.T) >= 0):
                raise ValueError("State covariance matrix R @ R.T must be positive semi-definite.")
            if not np.all(np.linalg.eigvals(Q @ Q.T) >= 0):
                raise ValueError("Observation covariance matrix Q @ Q.T must be positive semi-definite.")
            RR = R @ R.T
            QQ = Q @ Q.T
            return {'A': A, 'D': D, 'Q': QQ, 'R': RR}

        return update_state_space

    def define_parameter(self, definition: dict):
        for param in definition:
            if param not in self.parameters:
                raise ValueError(f"Derived parameter '{param}' not in parameters {list(self.parameters.keys())}.")
        temp_params = self.parameters.copy()
        for param, expr in definition.items():
            try:
                tree = ast.parse(expr, mode='eval')
                param_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                for p in param_names:
                    if p not in self.parameters:
                        raise ValueError(f"Parameter '{p}' in expression for '{param}' not in parameters {list(self.parameters.keys())}.")
            except SyntaxError:
                raise ValueError(f"Invalid expression for {param}: {expr}")
        for param, expr in definition.items():
            try:
                value = eval(expr, {"__builtins__": {}}, temp_params)
                temp_params[param] = value
            except Exception as e:
                raise ValueError(f"Error evaluating expression for {param}: {expr}. Error: {str(e)}")
        for param, expr in definition.items():
            value = temp_params[param]
            self.parameters[param] = value
            self._predef= definition
            self._predef_expressions[param]=expr
            self._derived_params[param] = value
    def _set_priors(self, priors: Union[Dict[str, tuple], List[tuple]], bounds: List[tuple]):
        """
        Set priors for Bayesian estimation and return a prior function using make_prior_function.

        Parameters:
        - priors: Dictionary or list of tuples specifying prior distributions.
                Dict format: {'param': ('dist_name', {'param1': value1, ...}), ...}
                List format: [('param', ('dist_name', {'param1': value1, ...})), ...]
        - bounds: List of (lower, upper) tuples for each free parameter.

        Returns:
        - Callable: Prior function that computes log-prior probability for a parameter vector.
        """
        if self.technique != 'Bayesian':
            raise ValueError("Priors can only be set for Bayesian estimation.")

        # Convert list to dict for uniform handling
        if isinstance(priors, list):
            priors = {param: dist_info for param, dist_info in priors}
        elif not isinstance(priors, dict):
            raise ValueError("Priors must be a dictionary or list of tuples.")

        # Get free parameters (exclude fixed and derived)
        param_names = [p for p in self.parameters.keys(
        ) if p not in self.fixed_params and p not in self._derived_params]
        if len(param_names) != len(priors):
            raise ValueError(
                f"Priors must be specified for all {len(param_names)} free parameters: {param_names}.")
        if len(param_names) != len(bounds):
            raise ValueError(
                f"Bounds must match the number of free parameters: {len(param_names)}.")
        supported_dists = {'gamma', 'beta', 'norm', 'uniform'}
        for param, (dist_name, dist_params) in priors.items():
            if param not in self.parameters:
                raise ValueError(
                    f"Parameter {param} not in self.parameters: {self.parameters.keys()}.")
            if param in self.fixed_params:
                raise ValueError(
                    f"Parameter {param} is fixed and cannot have a prior.")
            if param in self._derived_params:
                raise ValueError(
                    f"Parameter {param} is derived and cannot have a prior.")
            if dist_name not in supported_dists:
                raise ValueError(
                    f"Distribution {dist_name} not supported. Use: {supported_dists}.")
            if not isinstance(dist_params, dict):
                raise ValueError(
                    f"Distribution parameters for {param} must be a dictionary.")
        bounds_dict = {param: bound for param,
                       bound in zip(param_names, bounds)}
        for param, (lb, ub) in bounds_dict.items():
            if not isinstance(lb, (int, float)) or not (isinstance(ub, (int, float)) or ub == float('inf')):
                raise ValueError(
                    f"Bounds for parameter {param} must be numeric or inf for upper bound: ({lb}, {ub}).")
            if not np.isinf(ub) and lb >= ub:
                raise ValueError(
                    f"Lower bound {lb} must be less than upper bound {ub} for {param}.")
        prior = make_prior_function(
            param_names=param_names, priors=priors, bounds=bounds, verbose=True)
        self.prior_fn = prior
        return prior

    def calibrate_params(self, fixed_params:List[set]):
            for param, value in fixed_params:
                if param in self.parameters:
                    self.parameters[param] = value
                    self.fixed_params[param] = value
                else:
                    raise KeyError(f"Parameter '{param}' not in model parameters.")
            return self.fixed_params
        
    def fit(self, Lower_bound: list, Upper_bound: list, prior_specs: dict = None, seed: int = 1, T0: float = 5, rt: float = 0.9, nt: int = 2, ns: int = 2,
            pop_size: int = 50, n_gen: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.1, elite_frac: float = 0.1, stand_div: list = None,
            verbose: bool = True, tol: float = 1e-6, n_iter: int = 10000, burn_in: int = 100, thin: int = 10):
        results = None
        s_e_e_d = seed if seed > 0 else 1
        prior_specs = prior_specs or {}
        parameters = self.parameters.copy()
        param_names = [k for k in parameters.keys()if k not in self._derived_params.keys() and k not in self.fixed_params.keys()]
        initial_params = [parameters[name] for name in param_names]
        print(initial_params)
        self.validate_entries_()
        if self.validated_entries_:
            update_state_space = self._make_state_space_updater(self.parameters)
            self.state_updater=update_state_space
        bounds_set = [(lb, ub) for lb, ub in zip(Lower_bound, Upper_bound)]
        match self.technique:
            case 'MLE':
                if self.optimizer.lower() in ['sa', 'ga', 'trust_constr', 'bfgs', 'powell']:
                    if self.optimizer.lower() == 'sa':
                        results = simulated_annealing_kalman(y=self.data,x0=initial_params, lb=Lower_bound, ub=Upper_bound,param_names=param_names,fixed_params=self.fixed_params,update_state_space=update_state_space, seed=s_e_e_d, T0=T0, rt=rt, nt=nt, ns=ns)
                    elif self.optimizer.lower() == 'ga':
                        results = genetic_algorithm_kalman(self.data, initial_params, Lower_bound, Upper_bound, param_names, self.fixed_params, update_state_space, pop_size=pop_size,
                                                           n_gen=n_gen, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elite_frac=elite_frac, seed=s_e_e_d, verbose=verbose)
                    elif self.optimizer.lower() in ['trust_constr', 'bfgs', 'powell']:
                        try:
                            def obj_func(params): return kalman_objective(
                                params, self.fixed_params, param_names, self.data, update_state_space)
                            results = minimize(
                                obj_func, initial_params, method=self.optimizer, bounds=bounds_set)
                        except Exception as e:
                            error_result = {
                                'x': None, 'fun': None, 'nfev': None, 'message': f'GA Kalman failed: {str(e)}'}
                            print(
                                f"Error in genetic_algorithm_kalman: {e}, returning: {error_result}")
            case 'Bayesian':
                logger.info(f'Starting bayesian estimation')
                prior = self._set_priors(priors=prior_specs, bounds=bounds_set)
                sigma = compute_proposal_sigma(
                    len(bounds_set), Lower_bound, Upper_bound, base_std=stand_div)
                results = rwm_kalman(self.data, initial_params, Lower_bound, Upper_bound, param_names, self.fixed_params,
                                     update_state_space, n_iter=n_iter, burn_in=burn_in, thin=thin, sigma=sigma, seed=s_e_e_d, prior=prior)
        self.result = results
    def summary(self):
        """
        Generate a summary table and plots for model results.
        
        """
        if not hasattr(self, 'result') or self.result is None:
            raise ValueError("Model must be fitted before generating summary.")
        param_names = [k for k in self.parameters.keys() if k not in self._derived_params and k not in self.fixed_params]
        n_params = len(param_names)
        T = self.data.shape[1]
        log_lik = None
        aic = bic = hqic = None
        table_data = []
        if self.technique == 'MLE':
            x_opt = self.result.get('x')
            log_lik = self.result.get('fun') if self.result.get('fun') is not None else np.nan
            nfev = self.result.get('nfev', self.result.get('N_FUNC_EVALS', np.nan))
            message = self.result.get('message', 'No message available')
            if not np.isnan(log_lik):
                aic = 2 * n_params - 2 * log_lik
                bic = n_params * np.log(T) - 2 * log_lik
                hqic = 2 * n_params * np.log(np.log(T)) - 2 * log_lik
            if x_opt is not None:
                try:
                    def obj_func(params): return kalman_objective(params, self.fixed_params, param_names, self.data, self.state_updater)
                    stats = compute_stats(x_opt, log_lik, obj_func)
                    std_err = stats['std_err']
                    p_values = stats['p_values']
                    t_values = x_opt / std_err if np.all(std_err != 0) else np.array([np.nan] * len(x_opt))
                    for i, param in enumerate(param_names):
                        signif = ''
                        if not np.isnan(p_values[i]):
                            if p_values[i] < 0.001:
                                signif = '***'
                            elif p_values[i] < 0.01:
                                signif = '**'
                            elif p_values[i] < 0.05:
                                signif = '*'
                        table_data.append({'Parameter': param,'Value': x_opt[i],'Std. Err.': std_err[i],'p-value': p_values[i],'t-value': t_values[i],'Significance': signif})
                except Exception as e:
                    logger.warning(f"Failed to compute stats: {str(e)}")
                    table_data = [{'Parameter': param, 'Value': x_opt[i], 'Std. Err.': np.nan, 'p-value': np.nan, 't-value': np.nan, 'Significance': ''} for i, param in enumerate(param_names)]
            
            print("=" * 60)
            print(f"{'MODEL':^60}")
            print("=" * 60)
            print(f"Model: {getattr(self, 'name', 'State Space Model')}")
            print("=" * 60)
            print(f"{f'Log-Likelihood: {log_lik:.4f}' if not np.isnan(log_lik) else 'Log-Likelihood: N/A':<30}"f"{f'AIC: {aic:.4f}' if aic is not None else 'AIC: N/A':<30}")
            print(f"{f'BIC: {bic:.4f}' if bic is not None else 'BIC: N/A':<30}"
                f"{f'HQIC: {hqic:.4f}' if hqic is not None else 'HQIC: N/A':<30}")
            print("=" * 60)
            print(f"{'Technique: Maximum Likelihood':<30}"
                f"{f'Optimizer: {self.optimizer}':<30}")
            print(f"{f'Number of Estimated Parameters: {n_params}':<30}"
                f"{f'Number of Function Evaluations: {nfev}':<30}")
            print(f"Message: {message}")
            print("=" * 60)
            if table_data:
                print("_" * 60)
                print(f"{'Parameter':<15} | {'Value':<10} | {'Std. Err.':<10} | {'p-value':<10} | {'t-value':<10} | {'Sig':<5}")
                print("_" * 60)
                for row in table_data:
                    print(f"{row['Parameter']:<15} | {row['Value']:<10.4f} | {row['Std. Err.']:<10.4f} | "
                        f"{row['p-value']:<10.4f} | {row['t-value']:<10.4f} | {row['Significance']:<5}")
                print("-" * 60)
            print("=" * 60)
        elif self.technique == 'Bayesian':
            samples = self.result.get('samples')
            log_posterior = self.result.get('log_posterior', np.nan)
            acceptance_rate = self.result.get('acceptance_rate', np.nan)
            mean_params = self.result.get('mean_posterior_parameters')
            std_params = self.result.get('std_posterior_parameters')
            message = self.result.get('message', 'No message available')
            if not np.isnan(log_posterior):
                aic = 2 * n_params - 2 * log_posterior
                bic = n_params * np.log(T) - 2 * log_posterior
                hqic = 2 * n_params * np.log(np.log(T)) - 2 * log_posterior
            if mean_params is not None and std_params is not None:
                for i, param in enumerate(param_names):
                    table_data.append({
                        'Parameter': param,
                        'Mean': mean_params[i],
                        'Std. Dev.': std_params[i],
                        'Credible Interval (95%)': f"[{np.percentile(samples[:, i], 2.5):.4f}, {np.percentile(samples[:, i], 97.5):.4f}]",
                        'Significance': ''
                    })

            print("=" * 60)
            print(f"{'MODEL':^60}")  # Centers 'MODEL' in 60 characters
            print("=" * 60)
            print(f"Model: {getattr(self, 'name', 'State Space Model')}")
            print("=" * 60)
            print(f"{f'Log-Posterior: {log_posterior:.4f}' if not np.isnan(log_posterior) else 'Log-Posterior: N/A':<30}"
                f"{f'AIC: {aic:.4f}' if aic is not None else 'AIC: N/A':<30}")
            print(f"{f'BIC: {bic:.4f}' if bic is not None else 'BIC: N/A':<30}"
                f"{f'HQIC: {hqic:.4f}' if hqic is not None else 'HQIC: N/A':<30}")
            print("=" * 60)
            print(f"{'Technique: Bayesian':<30}"
                f"{f'Acceptance Rate: {acceptance_rate:.4f}' if not np.isnan(acceptance_rate) else 'Acceptance Rate: N/A':<30}")
            print(f"{f'Number of Estimated Parameters: {n_params}':<30}"
                f"{f'Message: {message}':<30}")
            print("=" * 60)
            if table_data:
                df = pd.DataFrame(table_data)
                print(df.to_string(index=False, float_format="%.4f"))
            print("=" * 60)
            if samples is not None:
                n_params = samples.shape[1]
                fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params), squeeze=False)
                for i, param in enumerate(param_names):
                    kde = gaussian_kde(samples[:, i])
                    x_range = np.linspace(np.min(samples[:, i]), np.max(samples[:, i]), 200)
                    axes[i, 0].plot(x_range, kde(x_range), label=f'{param} PDF', color='blue')
                    axes[i, 0].axvline(np.mean(samples[:, i]), color='red', linestyle='--', label='Mean')
                    axes[i, 0].axvline(np.percentile(samples[:, i], 2.5), color='green', linestyle=':', label='95% CI')
                    axes[i, 0].axvline(np.percentile(samples[:, i], 97.5), color='green', linestyle=':')
                    axes[i, 0].set_xlabel(param)
                    axes[i, 0].set_ylabel('Density')
                    axes[i, 0].set_title(f'Posterior Distribution of {param}')
                    axes[i, 0].legend()
                    axes[i, 0].grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        if self.result and (self.technique == 'MLE' and self.result.get('x') is not None) or (self.technique == 'Bayesian' and self.result.get('mean_posterior_parameters') is not None):
            full_params = self.parameters.copy()
            if self.technique == 'MLE':
                for param, value in zip(param_names, self.result['x']):
                    full_params[param] = value
            else:
                for param, value in zip(param_names, self.result['mean_posterior_parameters']):
                    full_params[param] = value
            if hasattr(self, '_predef') and self._predef:
                self.define_parameter(self._predef)
                for param in self._derived_params:
                    full_params[param] = self._derived_params[param]
            print(full_params)
            kalman_smooth(self.data, full_params,self.state_updater, plot=True)
    def predict(self, steps: int, test_data: Optional[np.ndarray] = None, plot: bool = False):
        """
        Generate predictions for the next 'steps' time points and evaluate against test_data.

        Parameters:
        -----------
        steps : int
            Number of future time points to predict.
        test_data : ndarray, optional
            Test data for evaluation (shape: m x T_test, where m matches self.data.shape[0]).
        plot : bool
            Whether to plot residuals (default: False).

        Returns:
        --------
        dict
            Predictions, smoothed states, and evaluation metrics (if test_data provided).
        """
        if not hasattr(self, 'result') or self.result is None:
            raise ValueError("Model must be fitted before generating predictions.")
        param_names = [k for k in self.parameters.keys() if k not in self._derived_params and k not in self.fixed_params]
        full_params = self.parameters.copy()
        if self.technique == 'MLE' and self.result.get('x') is not None:
            for param, value in zip(param_names, self.result['x']):
                full_params[param] = value
        elif self.technique == 'Bayesian' and self.result.get('mean_posterior_parameters') is not None:
            for param, value in zip(param_names, self.result['mean_posterior_parameters']):
                full_params[param] = value
        else:
            raise ValueError("No valid parameter estimates found in self.result.")
        if hasattr(self, '_predef') and self._predef:
            self.define_parameter(self._predef)
            for param in self._derived_params:
                full_params[param] = self._derived_params[param]
        ss_params = self.state_updater(full_params)
        A, D= ss_params['A'], ss_params['D'], ss_params['Q'], ss_params['R']
        try:
            kalman = Kalman(ss_params)
            smooth_result = kalman.smooth(self.data)
            Xsm = smooth_result['Xsm']
            last_state = Xsm[:, -1]
        except Exception as e:
            logger.error(f"Error running Kalman smoother on training data: {str(e)}")
            return {'predictions': None, 'smoothed_states': None, 'metrics': None, 'residuals': None}
        n_states = A.shape[0]
        T_pred = steps
        predicted_states = np.zeros((n_states, T_pred))
        predicted_states[:, 0] = A @ last_state 
        
        for t in range(1, T_pred):
            predicted_states[:, t] = A @ predicted_states[:, t-1]
        
        predictions = D @ predicted_states if D is not None else np.zeros((self.data.shape[0], T_pred))
        
        result = {'predictions': predictions, 'smoothed_states': predicted_states, 'metrics': None,'residuals': None}
        if test_data is not None:
            if test_data.shape[0] != self.data.shape[0]:
                raise ValueError(f"test_data must have {self.data.shape[0]} variables, got {test_data.shape[0]}.")
            if test_data.shape[1] < steps:
                logger.warning(f"test_data has fewer time points ({test_data.shape[1]}) than steps ({steps}). Using available data.")
                test_data = test_data[:, :min(steps, test_data.shape[1])]
                predictions = predictions[:, :test_data.shape[1]]
            residuals = test_data - predictions
        
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / test_data)) * 100 if np.all(test_data != 0) else np.nan
            
            result['metrics'] = {'MSE': mse,'RMSE': rmse,'MAE': mae,'MAPE': mape}
            result['residuals'] = residuals
            if plot:
                T_test = residuals.shape[1]
                time = np.arange(T_test)
                n_vars = residuals.shape[0]
                fig, axes = plt.subplots(n_vars, 1, figsize=(8, 4 * n_vars), squeeze=False)
                for i in range(n_vars):
                    axes[i, 0].plot(time, residuals[i, :], label='Residuals', color='purple')
                    axes[i, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
                    axes[i, 0].set_xlabel('Time')
                    axes[i, 0].set_ylabel(f'Residuals (Var {i+1})')
                    axes[i, 0].set_title(f'Residuals for Variable {i+1}')
                    axes[i, 0].legend()
                    axes[i, 0].grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        return result

    def evaluate(self, data: Optional[np.ndarray] = None, plot: bool = False):
        """
        Evaluate model residuals and diagnostics on training or test data.

        Parameters:
        -----------
        data : ndarray, optional
            Data to evaluate (defaults to self.data if None).
        plot : bool
            Whether to plot residuals (default: False).

        Returns:
        --------
        dict
            Residuals and diagnostics (mean, std, and metrics if data provided).
        """
        if not hasattr(self, 'result') or self.result is None:
            raise ValueError("Model must be fitted before evaluating residuals.")
        data = self.data if data is None else np.asarray(data)
        if data.shape[0] != self.data.shape[0]:
            raise ValueError(f"Data must have {self.data.shape[0]} variables, got {data.shape[0]}.")
        param_names = [k for k in self.parameters.keys() if k not in self._derived_params and k not in self.fixed_params]
        full_params = self.parameters.copy()
        if self.technique == 'MLE' and self.result.get('x') is not None:
            for param, value in zip(param_names, self.result['x']):
                full_params[param] = value
        elif self.technique == 'Bayesian' and self.result.get('mean_posterior_parameters') is not None:
            for param, value in zip(param_names, self.result['mean_posterior_parameters']):
                full_params[param] = value
        else:
            raise ValueError("No valid parameter estimates found in self.result.")
        if hasattr(self, '_predef') and self._predef:
            self.define_parameter(self._predef)
            for param in self._derived_params:
                full_params[param] = self._derived_params[param]
        smooth_result = kalman_smooth(data, full_params, self.state_updater ,plot=False)
        residuals = smooth_result['residuals']
        predictions = smooth_result['predictions']
        if residuals is None:
            logger.error("Failed to compute residuals in kalman_smooth.")
            return {'residuals': None, 'diagnostics': None, 'metrics': None}
        diagnostics = {'mean_residuals': np.mean(residuals, axis=1),'std_residuals': np.std(residuals, axis=1)}
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / data)) * 100 if np.all(data != 0) else np.nan
        metrics = {'MSE': mse,'RMSE': rmse,'MAE': mae,'MAPE': mape}
        if plot:
            T = data.shape[1]
            time = np.arange(T)
            n_vars = data.shape[0]
            fig, axes = plt.subplots(n_vars, 1, figsize=(8, 4 * n_vars), squeeze=False)
            for i in range(n_vars):
                axes[i, 0].plot(time, residuals[i, :], label='Residuals', color='purple')
                axes[i, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
                axes[i, 0].set_xlabel('Time')
                axes[i, 0].set_ylabel(f'Residuals (Var {i+1})')
                axes[i, 0].set_title(f'Residuals for Variable {i+1}')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        for i in range(n_vars):
            res = residuals[i, :]
            stat, p_value = shapiro(res)
            diagnostics[f'shapiro_stat_var{i+1}'] = stat
            diagnostics[f'shapiro_p_value_var{i+1}'] = p_value
            diagnostics[f'skewness_var{i+1}'] = skew(res)
            diagnostics[f'kurtosis_var{i+1}'] = kurtosis(res, fisher=True)
            lags=int(np.round(np.sqrt(self.data.shape[1])))
            acf_values, confint = acf(res, nlags=20, alpha=0.05, fft=True)
            diagnostics[f'acf_var{i+1}'] = acf_values
            diagnostics[f'acf_confint_var{i+1}'] = confint
            try:
                time_trend = np.arange(len(res))
                exog = sm.add_constant(time_trend)
                bp_test = het_breuschpagan(res, exog)
                diagnostics[f'breusch_pagan_stat_var{i+1}'] = bp_test[0]  # LM statistic
                diagnostics[f'breusch_pagan_p_value_var{i+1}'] = bp_test[1]  # LM p-value
            except Exception as e:
                logger.warning(f"Breusch-Pagan test failed for variable {i+1}: {str(e)}")
                diagnostics[f'breusch_pagan_stat_var{i+1}'] = np.nan
                diagnostics[f'breusch_pagan_p_value_var{i+1}'] = np.nan
        if plot:
            for i in range(n_vars):
                res = residuals[i, :]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                probplot(res, dist="norm", plot=ax1)
                ax1.set_title(f'Q-Q Plot for Residuals (Var {i+1})')
                ax1.set_xlabel('Theoretical Quantiles')
                ax1.set_ylabel('Sample Quantiles')
                ax1.grid(True, alpha=0.3)
                acf_values = diagnostics[f'acf_var{i+1}']
                confint = diagnostics[f'acf_confint_var{i+1}']
                lags = np.arange(len(acf_values))
                ax2.stem(lags, acf_values, use_line_collection=True)
                ax2.fill_between(lags, confint[:, 0] - acf_values, confint[:, 1] - acf_values, alpha=0.2)
                ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
                ax2.set_xlabel('Lag')
                ax2.set_ylabel('Autocorrelation')
                ax2.set_title(f'Autocorrelation of Residuals (Var {i+1})')
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        return {'residuals': residuals,'diagnostics': diagnostics,'metrics': metrics}