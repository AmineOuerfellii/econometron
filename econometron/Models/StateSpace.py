from typing import Union, Callable, Dict, List, Optional
import pandas as pd
import numpy as np
from econometron.Models.dynamicsge import linear_dsge
from econometron.utils.estimation.MLE import simulated_annealing_kalman, genetic_algorithm_kalman
from econometron.utils.estimation.Bayesian import rwm_kalman, compute_proposal_sigma, make_prior_function
from econometron.filters import kalman_objective, Kalman, kalman_smooth
import logging
from scipy.optimize import minimize
import ast
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SS_Model:
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, pd.Series], parameters: dict, P: np.ndarray, x0: np.ndarray, optimizer: str = 'L-BFGS-B', estimation_method: str = 'MLE', constraints: dict = None):
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
        self.parameters = parameters
        self.P = P
        self.x0 = x0
        self.optimizer = optimizer
        self.technique = estimation_method
        self.constraints = constraints
        self.A = None
        self.Q = None
        self.D = None
        self.R = None

    def validate_entries_(self):
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Initial covariance matrix P must be square.")
        if self.x0.shape[0] != self.P.shape[0]:
            raise ValueError("Initial state x0 must match P dimensions.")
        if self.model is None and (self.A is None or self.C is None or self.Q is None or self.R is None):
            raise ValueError(
                "Model OR transition matrix A, observation matrix C, Q, and R must be defined.")
        if self.model is not None and (self.A is not None or self.C is not None):
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
        if self.optimizer and self.optimizer not in optimizers_list:
            raise ValueError(f"Optimizer must be one of {optimizers_list}.")
        if self.constraints and self.optimizer != 'trust_constr':
            logger.info(
                f"The constraints aren't going to be taken into account since you've set your optimizer as {self.optimizer}")

    def set_transition_mat(self, A: Union[np.ndarray, np.matrix]):
        if self.A is None:
            if A.shape[0] != A.shape[1]:
                raise ValueError("Transition matrix A must be square.")
            self.A = np.array(A)
            return self.A

    def set_state_cov(self, Q: Union[np.ndarray, np.matrix], params: Optional[dict] = None):
        """Set the observation noise covariance matrix Q, optionally using parameters."""
        if self.A is None:
            raise ValueError(
                "Set transition matrix before setting observation matrix.")
        Q = np.asarray(Q)
        if Q.shape[0] != Q.shape[1] and Q.shape[0] != self.A.shape[0]:
            raise ValueError("Observation covariance matrix Q must be square.")
        if not np.all(np.linalg.eigvals(Q) >= 0):
            raise ValueError(
                "Observation covariance matrix Q must be positive semi-definite.")
        self.Q = Q
        return self.Q

    def set_obs_cov(self, R: Union[np.ndarray, np.matrix]):
        """Set the state noise covariance matrix R, optionally using parameters."""
        logger.info("R needs to be set a FUll covariance matrix")
        R = np.asarray(R)
        if R.shape[0] != R.shape[1] and R.shape[0] != self.D.shape[0]:
            raise ValueError("State covariance matrix R must be square.")
        if not np.all(np.linalg.eigvals(R) >= 0):
            logger.info("State covariance matrix R must be positive semi-definite.")
        self.R = R
        return self.R

    def set_Design_mat(self, D: Union[np.ndarray, np.matrix]):
        """Set the control matrix D, optionally using parameters."""
        if self.A is None:
            raise ValueError(
                "Set transition matrix before setting observation matrix.")
        D = np.asarray(D)
        if D.shape[1] != self.A.shape[0]:
            raise ValueError("Control matrix D must match state dimension.")
        self.D = D
        return self.D

    def define_parameter(self, definition: Dict[str, str]):
        """Update existing model parameters for custom optimizers using definition dictionary."""
        if self.technique not in ['MLE', 'Bayesian'] or self.optimizer not in ['sa', 'ga', None]:
            logger.warning(
                "Derived parameters are only used with custom optimizers (sa, ga, or Bayesian).")
            return
        self._initial_parameters = self.parameters.copy()
        self._derived_params = set()
        for param in definition:
            if param not in self.parameters:
                raise ValueError(f"Derived parameter {param} not in initial parameters {self.parameters.keys()}.")
        for param, expr in definition.items():
            try:
                tree = ast.parse(expr, mode='eval')
                param_names = {node.id for node in ast.walk(
                    tree) if isinstance(node, ast.Name)}
                for p in param_names:
                    if p not in self.parameters:
                        raise ValueError(f"Parameter {p} in expression for {param} not in initial parameters {self.parameters.keys()}.")
            except SyntaxError:
                raise ValueError(f"Invalid expression for {param}: {expr}")
        updated_params = self.parameters.copy()
        for param, expr in definition.items():
            try:
                value = eval(expr, {"__builtins__": {}}, self.parameters)
                updated_params[param] = value
                if value != self._initial_parameters[param]:
                    self._derived_params.add(param)
            except Exception as e:
                raise ValueError(
                    f"Error evaluating expression for {param}: {expr}. Error: {str(e)}")

        self.parameters.update(updated_params)

    def _make_state_space_updater(self, base_params: dict, solver: Optional[Callable] = None):
        """Create a state-space updater function."""
        def update_state_space(params):
            full_params = base_params.copy()
            full_params.update(params)
            if hasattr(self, 'definition') and self.definition:
                self.define_parameter(self.definition)
                full_params.update(self.parameters)
            if self.model is not None:
                D, A = solver(full_params)
            else:
                if any(x is None for x in [self.A, self.C, self.Q, self.R]):
                    raise ValueError(
                        "All state-space matrices (A, C, Q, R) must be set if no solver is provided.")
                A = self.A
                D = self.D if self.D is not None else np.zeros(
                    (self.x0.shape[0], 1))
            R = self.R
            C = self.C
            RR = R
            QQ = C
            return {'A': A, 'D': D, 'Q': QQ, 'R': RR}
        return update_state_space

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
        bounds_dict = {param: bound for param,bound in zip(param_names, bounds)}
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
    def calibrate_params(self, fixed_params: set):
        self.fixed_params = {param: self.parameters[param]for param in fixed_params if param in self.parameters}
    def fit(self, Lower_bound: list, Upper_bound: list, prior_specs: dict = None, seed: int = 1, T0: float = 5, rt: float = 0.9, nt: int = 2, ns: int = 2,
            pop_size: int = 50, n_gen: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.1, elite_frac: float = 0.1, stand_div: list = None,
            verbose: bool = True, tol: float = 1e-6, n_iter: int = 10000, burn_in: int = 100, thin: int = 10):
        results = None
        s_e_e_d = seed if seed > 0 else 42
        prior_specs = prior_specs or {}
        initial_params = self._initial_parameters
        param_names = [k for k in initial_params.keys()if k not in self._derived_params.keys() and k not in self.fixed_params.keys()]
        initial_params = [initial_params[name] for name in param_names]
        self._validate_entries_()
        if self.validated_entries_:
            update_state_space = self._make_state_space_updater(
                self.fixed_params, self.model.build_A, self.model.build_D, self.model.build_R, self.model.build_C, self.model.solve)
        bounds_set = [(lb, ub) for lb, ub in zip(Lower_bound, Upper_bound)]
        match self.technique:
            case 'MLE':
                if self.optimizer.lower() in ['sa', 'ga', 'trust_constr', 'bfgs', 'powell']:
                    if self.optimizer.lower() == 'sa':
                        results = simulated_annealing_kalman(self.data, initial_params, Lower_bound, Upper_bound,
                                                             param_names, self.fixed_params, update_state_space, seed=s_e_e_d, T0=T0, rt=rt, nt=nt, ns=ns)
                    elif self.optimizer.lower() == 'ga':
                        results = genetic_algorithm_kalman(self.data, initial_params, Lower_bound, Upper_bound, param_names, self.fixed_params, update_state_space, pop_size=pop_size,
                                                           n_gen=n_gen, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elite_frac=elite_frac, seed=s_e_e_d, verbose=verbose)
                    elif self.optimizer.lower() in ['trust_constr', 'bfgs', 'powell']:
                        try:
                            def obj_func(params): return kalman_objective(params, self.fixed_params, param_names, self.data, update_state_space)
                            results = minimize(obj_func, initial_params, method=self.optimizer, bounds=bounds_set)
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
        # Generate a summary of the model's parameters and state
        pass

    def predict(self, steps: int):
        # Generate predictions for the next 'steps' time points
        pass
