from typing import Union, Dict, List, Optional ,Callable
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
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, pd.Series], parameters: dict, model: linear_dsge = None, optimizer: str = 'L-BFGS-B', estimation_method: str = 'MLE', constraints: dict = None):
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
    def validate_entries_(self):
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
            if hasattr(self, 'definition') and self.definition:
                self.define_parameter(self.definition)
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
        
    def fit(self, Lower_bound: list, Upper_bound: list, prior_specs: dict = None, seed: int = 42, T0: float = 5, rt: float = 0.9, nt: int = 2, ns: int = 2,
            pop_size: int = 50, n_gen: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.1, elite_frac: float = 0.1, stand_div: list = None,
            verbose: bool = True, tol: float = 1e-6, n_iter: int = 10000, burn_in: int = 100, thin: int = 10):
        results = None
        s_e_e_d = seed if seed > 0 else 42
        prior_specs = prior_specs or {}
        parameters = self.parameters.copy()
        param_names = [k for k in parameters.keys()if k not in self._derived_params.keys() and k not in self.fixed_params.keys()]
        initial_params = [parameters[name] for name in param_names]
        print(initial_params)
        self.validate_entries_()
        if self.validated_entries_:
            update_state_space = self._make_state_space_updater(self.parameters)
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
        # Generate a summary of the model's parameters and state
        pass

    def predict(self, steps: int):
        # Generate predictions for the next 'steps' time points
        pass
