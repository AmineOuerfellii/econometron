# # ###########################
# Beginning with imports
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple, Any
from sympy import Symbol, Matrix, collect, S, log
import re
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import ordqz
import warnings
from econometron.utils.projection import ProjectionSolver
########
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Callable
####################
__all__ = ['linear_dsge', 'nonlinear_dsge']
####
# Time symbol
t = Symbol('t', integer=True)

# Transformations for parsing equations
_transformations = standard_transformations + \
    (implicit_multiplication_application,)

#################


class linear_dsge():

    def __init__(self, equations=None, variables=None, exo_states=None, endo_states=None, parameters=None, approximation=None, normalize=None, shocks=None):
        self.equations_list = equations
        self.names = {'variables': variables}
        self.variables = variables
        self.exo_states = exo_states
        self.endo_states = endo_states
        self.parameters = parameters
        if approximation in ['log', 'log_linear']:
            self.approximation = 'log_linear'
        else:
            self.approximation = 'linear'
        self.normalize = normalize if normalize is not None else {}
        ####################
        self.shocks = shocks
        ######################
        self.endo_states = endo_states if endo_states is not None else []
        self.exo_states = exo_states if exo_states is not None else []
        self.states = self.exo_states+self.endo_states
        self.controls = [
            var for var in self.variables if var not in self.states]
        ######################
        self.n_vars = len(self.variables)
        self.n_equations = len(self.equations_list)
        self.n_states = len(self.states)
        self.n_exo_states = len(exo_states) if exo_states else None
        self.n_endo_states = len(endo_states) if endo_states else None
        self.n_controls = len(self.controls)
        #####################
        self.steady_state = None
        self.f = None
        self.p = None
        self.c = None
        ###################
        self.stoch_simulated = None
        ###################
        self.irfs = None
        ###################
        # self.shock_variances = shock_variances or {shock: 0.01**2 for shock in shock_names}
        ###################
        self.approximated = None
        self.solved = None
        #########################################################################################

    def validate_entries(self):
        # Check equations
        if not self.equations_list or not isinstance(self.equations_list, list):
            raise ValueError("Equations must be provided as a non-empty list.")
        for eq in self.equations_list:
            if not isinstance(eq, str):
                raise ValueError(
                    f"Each equation must be a string. Invalid entry: {eq}")
            # Check for valid equation format
            if '=' not in eq:
                raise ValueError(
                    f"Invalid equation format: {eq}. Each equation must contain an '=' sign.")
        if self.equations_list == []:
            raise ValueError("Equations must be provided as a non-empty list.")
        # Check variables
        if not self.variables or not isinstance(self.variables, list):
            raise ValueError("Variables must be provided as a non-empty list.")
        # Check parameters
        if not self.parameters or not isinstance(self.parameters, dict):
            raise ValueError(
                "Parameters must be provided as a non-empty dictionary.")
        # Check shocks
        if self.shocks is None:
            warnings.warn(
                "No shocks specified. Model may be deterministic.", stacklevel=2)
        elif not isinstance(self.shocks, list):
            raise ValueError("Shocks must be provided as a list.")
        # Check exo_states and endo_states
        if self.exo_states is None:
            warnings.warn("Exogenous states not specified.", stacklevel=2)
        elif not isinstance(self.exo_states, list):
            raise ValueError("Exogenous states must be a list.")
        if self.endo_states is None:
            warnings.warn("Endogenous states not specified.", stacklevel=2)
        elif not isinstance(self.endo_states, list):
            raise ValueError("Endogenous states must be a list.")
        # Check for duplicate variable names
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(
                "Duplicate variable names found in variables list.")
        # Check for missing variables in equations
        missing_vars = [var for var in self.variables if not any(
            var in eq for eq in self.equations_list)]
        if missing_vars:
            warnings.warn(
                f"Variables {missing_vars} do not appear in any equation.", stacklevel=2)
        # Check approximation
        if self.approximation not in ['linear', 'log_linear']:
            warnings.warn(
                f"Unknown approximation type: {self.approximation}", stacklevel=2)
        # Check normalize
        if self.normalize and not isinstance(self.normalize, dict):
            warnings.warn(
                "Normalize should be a dictionary mapping variable names to values.", stacklevel=2)
        return True

    def set_new_pramaters(self, params):
        try:
            self.parameters.update(params)
            print('parameters updated:', self.parameters)
        except Exception as e:
            print(e)
            warnings.warn("parameters are unupdated")

    def set_initial_guess(self, initial_guess):
        """
        this function sets the initial guess for the model
        """
        if not isinstance(initial_guess, list):
            raise ValueError("Initial guess must be a list.")
        if len(initial_guess) != len([var for var in self.variables if var not in self.exo_states]):
            raise ValueError(
                "Initial guess must match the number of variables.")
        self.initial_guess = np.array(initial_guess)

    def _parse_equations(self, eq):
        """
        model._parse_equations(eq), is the function resposable of parsing the equations of a certain dynmic model
        This pacakge enables Users to write function as they are in the model , no traditional techniques are used
        so a user can write his model a a "String" and this function is the parser

        parameters:

        eq: equations

        """
        local_dict = {}
        for var in self.variables:
            local_dict[f"{var}_t"] = Symbol(f"{var}_t")
            local_dict[f"{var}_tp1"] = Symbol(f"{var}_tp1")
            local_dict[f"{var}_tm1"] = Symbol(f"{var}_tm1")
        for shock in self.shocks:
            local_dict[f"{shock}"] = Symbol(f"{shock}")
            local_dict[f"{shock}_t"] = Symbol(f"{shock}_t")
        for param in self.parameters:
            local_dict[param] = Symbol(param)

        eq_normalized = re.sub(
            r"[{\(]t([+-]1)?[}\)]", lambda m: {None: "t", "+1": "tp1", "-1": "tm1"}[m.group(1)], eq)
        if '=' in eq_normalized:
            left, right = eq_normalized.split('=')
            left_expr = parse_expr(
                left, local_dict=local_dict, transformations='all')
            right_expr = parse_expr(
                right, local_dict=local_dict, transformations='all')
            expr = left_expr - right_expr
        else:
            expr = parse_expr(
                eq_normalized, local_dict=local_dict, transformations='all')

        tp1_terms = S.Zero
        t_terms = S.Zero
        shock_terms = S.Zero
        all_vars = set(local_dict.keys())
        expr = expr.expand()
        for term in expr.as_ordered_terms():
            term_str = str(term)
            term_symbols = term.free_symbols
            is_constant = term_symbols and all(
                str(sym) in self.parameters for sym in term_symbols)
            has_shock = any(f"{shock}_t" in term_str for shock in self.shocks)
            has_tp1 = 'tp1' in term_str
            has_t = any(
                f"{var}_t" in term_str for var in self.variables) and not has_tp1
            if has_tp1 and not has_t and not has_shock:
                tp1_terms += term
            elif has_t and not has_tp1 and not has_shock:
                t_terms += term
            elif is_constant or has_shock:
                shock_terms += term
            else:
                coeff_dict = collect(term, [local_dict[f"{var}_tp1"] for var in self.variables] +
                                     [local_dict[f"{var}_t"] for var in self.variables] +
                                     ([local_dict[f"{shock}"] for shock in self.shocks] or [local_dict[f"{shock}_t"]for shock in self.shocks]), evaluate=False)

                for sym, coeff in coeff_dict.items():
                    sym_str = str(sym)
                    if 'tp1' in sym_str:
                        tp1_terms += coeff * sym
                    elif any(f"{shock}_t" in sym_str for shock in self.shocks):
                        shock_terms += coeff * sym
                    else:
                        t_terms += coeff * sym
                if Symbol('1') in coeff_dict:
                    shock_terms += coeff_dict[Symbol('1')]

        return -tp1_terms, t_terms, shock_terms

    def equations(self, vars_t_plus_1, vars_t, parameters):
        """
        Evaluate the model equations for compute_ss

        parameters :

        vars_t_plus_1
        vars_t
        parameters

        return:

        residuals
        """
        # Convert inputs to numpy arrays
        if isinstance(vars_t_plus_1, pd.Series):
            vars_t_plus_1 = vars_t_plus_1.values
        if isinstance(vars_t, pd.Series):
            vars_t = vars_t.values
        vars_t_plus_1 = np.array(vars_t_plus_1, dtype=float)
        vars_t = np.array(vars_t, dtype=float)

        residuals = []
        subs = {}
        for i, var in enumerate(self.variables):
            subs[Symbol(f"{var}_t")] = vars_t[i]
            subs[Symbol(f"{var}_tp1")] = vars_t_plus_1[i]
            subs[Symbol(f"{var}_tm1")] = vars_t[i]
        for shock in self.shocks:
            subs[Symbol(f"{shock}")] = parameters.get(shock, 0.0)
        subs.update({Symbol(k): float(v) for k, v in parameters.items()})

        for i, eq in enumerate(self.equations_list):
            tp1_terms, t_terms, shock_terms = self._parse_equations(eq)
            # print(f"Parsed equation {i+1}: t+1={tp1_terms}, t={t_terms}, shocks={shock_terms}")

            # Residual: LHS_{t+1} - RHS_t - shocks
            residual = (tp1_terms - t_terms - shock_terms).subs(subs)
            try:
                residual_value = float(residual.evalf().as_real_imag()[0])
                residuals.append(residual_value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Cannot convert residual to float for equation '{eq}': {residual}")

        return np.array(residuals)

    def compute_ss(self, guess=None, method='fsolve', options=None):
        if options is None:
            options = {}
        endogenous_vars = [var for var in self.variables if var not in self.exo_states]
        n_endogenous_vars = len(endogenous_vars)
        exo_ss_values = {}
        for var in self.exo_states:
            if hasattr(self, 'normalize') and var in self.normalize:
                # specified override always takes priority
                exo_ss_values[var] = float(self.normalize[var])
            elif self.steady_state is not None and var in self.steady_state:
                exo_ss_values[var] = float(self.steady_state[var])
            else:
                # Default fallback
                exo_ss_values[var] = 1.0 if self.approximation == "log_linear" else 0.0
        if guess is None:
            if self.initial_guess is not None:
                guess = self.initial_guess
            else:
                guess = np.ones(n_endogenous_vars)
                print("No initial guess provided. Using ones as default.")
        else:
            guess = np.array(guess, dtype=float)

            if len(guess) != n_endogenous_vars:
                raise ValueError(
                    f"Initial guess must have length {n_endogenous_vars}.")

        def ss_fun(variables):
            full_vars = []
            var_dict = {var: val for var, val in zip(endogenous_vars, variables)}
            for var in self.variables:
                if var in endogenous_vars:
                    full_vars.append(var_dict[var])
                else:
                    full_vars.append(exo_ss_values[var])
            residuals = self.equations(full_vars, full_vars, self.parameters)
            if len(residuals) != self.n_equations:
                raise ValueError(
                    f"Expected {self.n_equations} residuals, got {len(residuals)}.")
            # Return residuals for endogenous equations only for optimization
            endo_indices = []
            for i, eq in enumerate(self.equations_list):
                for var in endogenous_vars:
                    if re.search(rf"\b{var}(_t|_tp1)?\b", eq):
                        endo_indices.append(i)
                        break
            return residuals[endo_indices]

        if method == 'fsolve':
            steady_state = fsolve(ss_fun, guess, **options)
        else:
            raise ValueError("Only 'fsolve' is implemented.")

        steady_state_dict = {var: val for var,
                             val in zip(endogenous_vars, steady_state)}
        for var in self.variables:
            if var not in endogenous_vars:
                steady_state_dict[var] = exo_ss_values[var]
        self.steady_state = pd.Series(steady_state_dict, index=self.variables)

        # Compute residuals for all equations
        full_vars = [steady_state_dict[var] for var in self.variables]
        residuals = self.equations(full_vars, full_vars, self.parameters)
        print("Steady-state residuals:", residuals)
        if np.any(np.abs(residuals) > 1e-8):
            print("Warning: Large steady-state residuals detected.")

        return self.steady_state

    def _reorder_variables(self):
        """
        Reorder variables to follow [states, control] where states = exo_states + endo_states.
        Handles cases where exo_states or endo_states are None or empty.
        Returns the reordered variable list.
        """
        reordered = self.states + self.controls
        if set(reordered) != set(self.variables):
            raise ValueError(
                f"Reordered variables do not match original set: "f"reordered={reordered}, original={self.variables}")
        return reordered

    def _Analytical_jacobians(self, debug=False):
        """
        Compute Jacobians A, B, and C for the DSGE model A E_t[y_{t+1}] = B y_t + C epsilon_t.
        Variables are ordered as [exo_states, endo_states, controls].

        Returns:
            A (ndarray): Jacobian with respect to y_{t+1}.
            B (ndarray): Jacobian with respect to y_t.
            C (ndarray): Jacobian with respect to shocks epsilon_t.
        """
        # Use the same variable ordering as numerical method
        ordered_vars = self._reorder_variables()
        if debug:
            print("Ordered variables:", ordered_vars)

        # Check for steady states
        if self.steady_state is None:
            raise ValueError(
                "Steady state not computed. Call compute_ss() first")

        # Initialize matrices
        n_eqs = len(self.equations_list)
        n_vars = len(ordered_vars)
        n_shocks = len(self.shocks) if self.shocks else 0

        A = np.zeros((n_eqs, n_vars))
        B = np.zeros((n_eqs, n_vars))
        C = np.zeros((n_eqs, n_shocks))

        # Create symbols for variables
        vars_t = [Symbol(f"{var}_t") for var in ordered_vars]
        vars_tp1 = [Symbol(f"{var}_tp1") for var in ordered_vars]
        shock_symbols = [Symbol(shock)
                         for shock in self.shocks] if self.shocks else []

        # Steady state substitution dictionary
        subs = {}
        for var in ordered_vars:
            subs[Symbol(f"{var}_t")] = self.steady_state[var]
            subs[Symbol(f"{var}_tp1")] = self.steady_state[var]
            subs[Symbol(f"{var}_tm1")] = self.steady_state[var]

        for shock in self.shocks if self.shocks else []:
            subs[Symbol(shock)] = 0.0
            subs[Symbol(f"{shock}_t")] = 0.0

        subs.update({Symbol(k): float(v) for k, v in self.parameters.items()})

        if debug:
            print("Substitution dictionary:", subs)

        # Process each equation
        for eq_idx, eq in enumerate(self.equations_list):
            if debug:
                print(f"\nProcessing equation {eq_idx}: {eq}")

            # Parse the equation into components
            tp1_terms, t_terms, shock_terms = self._parse_equations(eq)

            if debug:
                print(f"  tp1_terms: {tp1_terms}")
                print(f"  t_terms: {t_terms}")
                print(f"  shock_terms: {shock_terms}")

            if self.approximation == 'log_linear':
                # For log-linear approximation
                if np.any(np.isclose([self.steady_state[var] for var in ordered_vars], 0)):
                    raise ValueError(
                        "Steady state contains zeros; cannot compute log-linear Jacobians.")

                # Create log variables
                log_vars_t = [Symbol(f"log_{var}_t") for var in ordered_vars]
                log_vars_tp1 = [Symbol(f"log_{var}_tp1")
                                for var in ordered_vars]
                log_shocks = [Symbol(f"log_{shock}")
                              for shock in self.shocks] if self.shocks else []

                # Build the linearized equation around steady state
                expr = tp1_terms - t_terms - shock_terms

                # Handle log expressions or linearize around steady state
                if "log(" in str(expr):
                    # Direct log substitution
                    subs_log = {}
                    for var, log_var, log_var_tp1 in zip(ordered_vars, log_vars_t, log_vars_tp1):
                        subs_log[log(Symbol(f"{var}_t"))] = log_var
                        subs_log[log(Symbol(f"{var}_tp1"))] = log_var_tp1
                    for shock, log_shock in zip(self.shocks if self.shocks else [], log_shocks):
                        subs_log[Symbol(shock)] = log_shock
                        subs_log[Symbol(f"{shock}_t")] = log_shock
                    expr = expr.subs(subs_log)
                else:
                    # Linearize around steady state
                    linear_expr = expr.subs(subs)

                    # Add first-order terms for current period variables
                    for j, var in enumerate(ordered_vars):
                        deriv = expr.diff(Symbol(f"{var}_t")).subs(subs)
                        linear_expr += deriv * \
                            self.steady_state[var] * log_vars_t[j]

                    # Add first-order terms for forward period variables
                    for j, var in enumerate(ordered_vars):
                        deriv = expr.diff(Symbol(f"{var}_tp1")).subs(subs)
                        linear_expr += deriv * \
                            self.steady_state[var] * log_vars_tp1[j]

                    # Add shock terms
                    for j, shock in enumerate(self.shocks if self.shocks else []):
                        deriv = expr.diff(Symbol(shock)).subs(subs)
                        linear_expr += deriv * log_shocks[j]

                    expr = linear_expr

                # Compute Jacobians for log-linear case
                for j, log_var_tp1 in enumerate(log_vars_tp1):
                    coeff = expr.diff(log_var_tp1)
                    try:
                        A[eq_idx, j] = float(coeff.subs(
                            {sym: 0 for sym in log_vars_t + log_vars_tp1 + log_shocks}))
                    except (TypeError, ValueError):
                        A[eq_idx, j] = 0.0

                for j, log_var_t in enumerate(log_vars_t):
                    coeff = expr.diff(log_var_t)
                    try:
                        B[eq_idx, j] = float(coeff.subs(
                            {sym: 0 for sym in log_vars_t + log_vars_tp1 + log_shocks}))
                    except (TypeError, ValueError):
                        B[eq_idx, j] = 0.0

                for j, log_shock in enumerate(log_shocks):
                    coeff = expr.diff(log_shock)
                    try:
                        C[eq_idx, j] = float(coeff.subs(
                            {sym: 0 for sym in log_vars_t + log_vars_tp1 + log_shocks}))
                    except (TypeError, ValueError):
                        C[eq_idx, j] = 0.0

            else:
                # Linear approximation
                # Compute A matrix (derivatives w.r.t. y_{t+1})
                for j, var_tp1 in enumerate(vars_tp1):
                    coeff = tp1_terms.diff(
                        var_tp1) if tp1_terms != S.Zero else S.Zero
                    try:
                        A[eq_idx, j] = float(coeff.subs(
                            subs)) if coeff != S.Zero else 0.0
                    except (TypeError, ValueError):
                        A[eq_idx, j] = 0.0

                # Compute B matrix (derivatives w.r.t. y_t)
                for j, var_t in enumerate(vars_t):
                    coeff = t_terms.diff(
                        var_t) if t_terms != S.Zero else S.Zero
                    try:
                        B[eq_idx, j] = float(coeff.subs(
                            subs)) if coeff != S.Zero else 0.0
                    except (TypeError, ValueError):
                        B[eq_idx, j] = 0.0

                # Compute C matrix (derivatives w.r.t. shocks)
                for j, shock_sym in enumerate(shock_symbols):
                    coeff = shock_terms.diff(
                        shock_sym) if shock_terms != S.Zero else S.Zero
                    try:
                        C[eq_idx, j] = float(coeff.subs(
                            subs)) if coeff != S.Zero else 0.0
                    except (TypeError, ValueError):
                        C[eq_idx, j] = 0.0

        # Create equation-to-variable mapping to match numerical method
        eq_to_var = {}
        remaining_eqs = list(range(len(self.equations_list)))

        # First pass: match equations with forward-looking variables
        for var in ordered_vars:
            for i, eq in enumerate(self.equations_list):
                if i not in remaining_eqs:
                    continue
                tp1_terms, _, _ = self._parse_equations(eq)
                if Symbol(f"{var}_tp1") in tp1_terms.free_symbols:
                    if var not in eq_to_var.values():
                        eq_to_var[i] = var
                        remaining_eqs.remove(i)
                        break

        # Second pass: match remaining equations with current period variables
        for var in ordered_vars:
            if var not in eq_to_var.values():
                for i, eq in enumerate(self.equations_list):
                    if i not in remaining_eqs:
                        continue
                    _, t_terms, _ = self._parse_equations(eq)
                    if Symbol(f"{var}_t") in t_terms.free_symbols:
                        eq_to_var[i] = var
                        remaining_eqs.remove(i)
                        break

        # Handle any remaining equations
        remaining_vars = [
            v for v in ordered_vars if v not in eq_to_var.values()]
        for i, eq_idx in enumerate(remaining_eqs):
            if i < len(remaining_vars):
                eq_to_var[eq_idx] = remaining_vars[i]

        if debug:
            print("Equation to variable mapping:", eq_to_var)

        # Create reordering index to match numerical method
        reorder_idx = [0] * len(self.equations_list)
        for eq_idx, var in eq_to_var.items():
            var_idx = ordered_vars.index(var)
            reorder_idx[eq_idx] = var_idx

        if debug:
            print("Reordering index:", reorder_idx)

        # Reorder rows to match numerical Jacobians
        A_reordered = np.zeros_like(A)
        B_reordered = np.zeros_like(B)
        C_reordered = np.zeros_like(C)

        for i, idx in enumerate(reorder_idx):
            A_reordered[idx, :] = A[i, :]
            B_reordered[idx, :] = B[i, :]
            C_reordered[idx, :] = C[i, :]

        if debug:
            print("Analytical Jacobian A (reordered):\n", A_reordered)
            print("Analytical Jacobian B (reordered):\n", B_reordered)
            print("Analytical Jacobian C (reordered):\n", C_reordered)
            if np.allclose(C_reordered, 0) and self.shocks:
                print("Warning: C matrix is all zeros. Check shock specifications.")
        # Return the reordered Jacobians
        A, B, C = A_reordered, B_reordered, C_reordered
        return A, B, C

    def _approx_fprime(self, x, f, epsilon=None):
        n = len(x)
        fx = f(x)
        m = len(fx)
        J = np.zeros((m, n))
        for i in range(n):
            eps = 1e-6 * max(1, abs(x[i]))
            x_eps = x.copy()
            x_eps[i] += eps
            J[:, i] = (f(x_eps) - fx) / eps
        return J

    def _Numerical_jacobians(self, debug=False):
        """
        Compute numerical Jacobians A, B, and C for the DSGE model A E_t[y_{t+1}] = B y_t + C epsilon_t
        using finite differences. Variables are ordered as [exo_states, endo_states, costates].
        Equations are reordered to match variable order using equation-to-variable mapping.
        Returns:
            A_num (ndarray): Numerical Jacobian with respect to y_{t+1}.
            B_num (ndarray): Numerical Jacobian with respect to y_t.
            C_num (ndarray): Numerical Jacobian with respect to shocks epsilon_t.
        """
        ordered_vars = self._reorder_variables()
        if debug:
            print("Reordered variables:", ordered_vars)

        # Equation-to-variable mapping
        eq_to_var = {}
        remaining_eqs = list(range(len(self.equations_list)))
        for i, eq in enumerate(self.equations_list):
            tp1_terms, _, _ = self._parse_equations(eq)
            for var in ordered_vars:
                if Symbol(f"{var}_tp1") in tp1_terms.free_symbols:
                    if var not in eq_to_var.values() and i in remaining_eqs:
                        eq_to_var[i] = var
                        remaining_eqs.remove(i)
                        break
            if i not in eq_to_var:
                _, t_terms, _ = self._parse_equations(eq)
                for var in ordered_vars:
                    if Symbol(f"{var}_t") in t_terms.free_symbols:
                        if var not in eq_to_var.values() and i in remaining_eqs:
                            eq_to_var[i] = var
                            remaining_eqs.remove(i)
                            break
        remaining_vars = [
            v for v in ordered_vars if v not in eq_to_var.values()]
        for i, eq_idx in enumerate(remaining_eqs):
            if i < len(remaining_vars):
                eq_to_var[eq_idx] = remaining_vars[i]

        if debug:
            print("Equation to variable mapping:", eq_to_var)

        # Create reordering index for equations
        reorder_idx = [0] * len(self.equations_list)
        for eq_idx, var in eq_to_var.items():
            var_idx = ordered_vars.index(var)
            reorder_idx[eq_idx] = var_idx
        if debug:
            print("Reordering index:", reorder_idx)

        # Steady state values in the reordered variable order
        e_s = np.array([self.steady_state[var]
                       for var in ordered_vars], dtype=np.float64)
        A_num = np.zeros((len(self.equations_list), len(ordered_vars)))
        B_num = np.zeros((len(self.equations_list), len(ordered_vars)))

        # Use shock names consistently
        shock_names = self.shocks if self.shocks else []
        C_num = np.zeros((len(self.equations_list), len(shock_names)))

        if not shock_names:
            print("Warning: No shocks identified. Check self.shocks:", self.shocks)

        # Map shock names to symbols
        shock_symbols = {shock: Symbol(shock) for shock in shock_names}

        # Exclude shock names from parameters to avoid collision
        parameters = {k: v for k, v in self.parameters.items()
                      if k not in shock_names}

        if self.approximation == 'log_linear':
            if np.any(np.isclose(e_s, 0)):
                raise ValueError(
                    "Steady state contains zeros; cannot compute log-linear Jacobians.")

            def psi(log_vars_fwd, log_vars_cur, log_shocks=None):
                vars_fwd = np.exp(log_vars_fwd)
                vars_cur = np.exp(log_vars_cur)
                shocks = log_shocks if log_shocks is not None else np.zeros(
                    len(shock_names))
                residuals = np.zeros(len(self.equations_list))

                for i, eq in enumerate(self.equations_list):
                    tp1_terms, t_terms, shock_terms = self._parse_equations(eq)
                    expr = tp1_terms + t_terms + shock_terms

                    if debug:
                        print(
                            f"Equation {i+1}: {eq}, shock_terms: {shock_terms}")

                    if "log(" in str(expr):
                        subs = {}
                        for j, var in enumerate(ordered_vars):
                            subs[log(Symbol(f"{var}_t"))] = log_vars_cur[j]
                            subs[log(Symbol(f"{var}_tp1"))] = log_vars_fwd[j]
                        for j, shock in enumerate(shock_names):
                            subs[Symbol(shock)] = shocks[j]
                        subs.update(parameters)
                    else:
                        subs = {
                            Symbol(f"{var}_t"): vars_cur[j] for j, var in enumerate(ordered_vars)}
                        subs.update(
                            {Symbol(f"{var}_tp1"): vars_fwd[j] for j, var in enumerate(ordered_vars)})
                        for j, shock in enumerate(shock_names):
                            subs[Symbol(shock)] = shocks[j]
                        subs.update(parameters)

                    if debug:
                        print(f"Equation {i+1} expr before subs: {expr}")

                    expr = expr.subs(subs)

                    try:
                        residuals[i] = float(expr)
                    except (ValueError, TypeError) as e:
                        if debug:
                            print(
                                f"Error evaluating equation {i+1}: {eq}, expr: {expr}, error: {e}")
                        residuals[i] = np.nan

                # Reorder residuals according to reorder_idx
                reordered_residuals = np.zeros_like(residuals)
                for i, idx in enumerate(reorder_idx):
                    reordered_residuals[idx] = residuals[i]

                if debug:
                    print(
                        f"Log-linear residuals (fwd={log_vars_fwd}, cur={log_vars_cur}, shocks={shocks}): {reordered_residuals}")
                return reordered_residuals

            log_ss = np.log(e_s)
            log_shocks_ss = np.zeros(len(shock_names))

            def psi_fwd(log_fwd): return psi(log_fwd, log_ss, log_shocks_ss)
            def psi_cur(log_cur): return psi(log_ss, log_cur, log_shocks_ss)
            def psi_shocks(log_shocks): return psi(log_ss, log_ss, log_shocks)

            A_num = self._approx_fprime(log_ss, psi_fwd)
            B_num = self._approx_fprime(log_ss, psi_cur)
            C_num = self._approx_fprime(log_shocks_ss, psi_shocks)

        else:
            def psi(vars_fwd, vars_cur, shocks=None):
                residuals = np.zeros(len(self.equations_list))
                shocks = shocks if shocks is not None else np.zeros(
                    len(shock_names))

                for i, eq in enumerate(self.equations_list):
                    tp1_terms, t_terms, shock_terms = self._parse_equations(eq)
                    if debug:
                        print(
                            f"Equation {i+1}: {eq}, shock_terms: {shock_terms}")

                    subs = {Symbol(f"{var}_t"): vars_cur[j]
                            for j, var in enumerate(ordered_vars)}
                    subs.update(
                        {Symbol(f"{var}_tp1"): vars_fwd[j] for j, var in enumerate(ordered_vars)})
                    for j, shock in enumerate(shock_names):
                        subs[Symbol(shock)] = shocks[j]

                    subs.update(parameters)

                    expr = tp1_terms - t_terms - shock_terms

                    if debug:
                        print(f"Equation {i+1} expr before subs: {expr}")

                    expr = expr.subs(subs)

                    try:
                        residuals[i] = float(expr)
                    except (ValueError, TypeError) as e:
                        if debug:
                            print(
                                f"Error evaluating equation {i+1}: {eq}, expr: {expr}, error: {e}")
                        residuals[i] = np.nan

                # Reorder residuals according to reorder_idx
                reordered_residuals = np.zeros_like(residuals)
                for i, idx in enumerate(reorder_idx):
                    reordered_residuals[idx] = residuals[i]

                if debug:
                    print(
                        f"Non-log-linear residuals (fwd={vars_fwd}, cur={vars_cur}, shocks={shocks}): {reordered_residuals}")
                return reordered_residuals

            def psi_fwd(fwd): return psi(fwd, e_s)
            def psi_cur(cur): return -psi(e_s, cur)
            def psi_shocks(shocks): return -psi(e_s, e_s, shocks)

            A_num = self._approx_fprime(e_s, psi_fwd)
            B_num = self._approx_fprime(e_s, psi_cur)
            C_num = self._approx_fprime(np.zeros(len(shock_names)), psi_shocks)

            if debug:
                for j, shock in enumerate(shock_names):
                    eps = 1e-6
                    shocks_pert = np.zeros(len(shock_names))
                    shocks_pert[j] = eps
                    residuals_pert = psi(e_s, e_s, shocks_pert)
                    residuals_base = psi(e_s, e_s, np.zeros(len(shock_names)))
                    deriv = (residuals_pert - residuals_base) / eps
                    print(
                        f"Shock {shock} perturbation: residuals_pert={residuals_pert}, residuals_base={residuals_base}")
                    print(f"Shock {shock} numerical derivative: {deriv}")

        self.A_num = A_num
        self.B_num = B_num
        self.C_num = C_num

        if debug:
            print("Numerical Jacobian A:\n", A_num)
            print("Numerical Jacobian B:\n", B_num)
            print("Numerical Jacobian C:\n", C_num)
            if np.allclose(C_num, 0) and shock_names:
                print(
                    "Warning: C_num matrix is all zeros despite shocks. Check equation specifications or _parse_equation logic.")

        return A_num, B_num, C_num

    def approximate(self, method=None, debug=False):
        """
        Approximates the RE model around its steady state using analytical or numerical methods.

        Parameters:
            method (str): 'analytical' or 'numerical' (default: 'analytical' if None)
            debug (bool): If True, prints intermediate steps for debugging

        Returns:
            tuple: (A, B, C) Jacobians for the system A E_t[y_{t+1}] = B y_t + C epsilon_t
        """
        if self.steady_state is None:
            raise ValueError(
                "Steady state not computed. Call compute_ss() first.")
        if self.approximated == True:
            warnings.warn("The system is already approximated.")
        # Default to analytical method if not specified
        method = method.lower() if method else 'analytical'
        if method == 'analytical':
            A, B, C = self._Analytical_jacobians(debug=debug)
        elif method == 'numerical':
            A, B, C = self._Numerical_jacobians(debug=debug)
        else:
            raise ValueError("Method must be 'analytical' or 'numerical'.")
        self.A = A
        self.B = B
        self.C = C
        self.approximated = True
        if debug:
            print(
                f"Approximation ({self.approximation}) completed with method: {method}")
            print(f"Jacobian A:\n{A}")
            print(f"Jacobian B:\n{B}")
            print(f"Jacobian C:\n{C}")
        return A, B, C

    def solve_RE_model(self, Parameters=None, debug=False):
        """
        Solves the rational expectations model A E_t[y_{t+1}] = B y_t + C epsilon_t.

        Parameters:
            Parameters (dict, optional): Model parameters to update before solving.
                                        If None, uses existing Jacobians (A, B, C).

        Returns:
            tuple: (P, Q) where P is the policy function (y_t = P s_t),
                  Q is the state transition (s_{t+1} = Q s_t + shocks).
        """
        if not self.approximated:
            raise ValueError(
                "Model not approximated. Call approximate() first.")
        if Parameters is not None:
            self.parameters = Parameters
            A, B, C = self._Analytical_jacobians(debug=debug)
        else:
            A, B, C = self.A, self.B, self.C
        if A is None or B is None or C is None:
            raise ValueError(
                "Jacobians A, B, C must be provided or computed via approximate().")
        if self.n_states == 0 or self.n_controls == 0:
            raise ValueError("Model must have states and controls defined.")
        # print('A',A)
        # print('B',B)

        def solve_klein(A, B, C, nk):
            """
            Solve using Klein's method (generalized Schur decomposition).
            Reference: Klein (2000) for linear rational expectations models.

            Args:
                A (ndarray): Jacobian matrix A
                B (ndarray): Jacobian matrix B
                C (ndarray): Jacobian matrix C
                nk (int): Number of state variables

            Returns:
                tuple: (F, P) where F is the control function, P is the state transition
            """
            n = A.shape[0]
            ns = C.shape[1] if C is not None else 0
            if debug:
                # Debugging information
                print(f"Model dimensions: n={n}, nk={nk}, ns={ns}")
                print(
                    f"Matrix shapes: A={A.shape}, B={B.shape}, C={C.shape if C is not None else None}")
                print(
                    f"Variable order: {self.variables[:nk]} (states) + {self.variables[nk:]} (controls)")
            try:
                S, T, alpha, beta, Q, Z = ordqz(
                    A, B, sort='ouc', output='complex')
            except Exception as e:
                raise ValueError(f"QZ decomposition failed: {e}")
            eigenvals = np.abs(beta / alpha)
            if debug:
                print(f"QZ eigenvalues (should have {nk} stable): {eigenvals}")
            z11 = Z[:nk, :nk]
            z21 = Z[nk:, :nk]
            s11 = S[:nk, :nk]
            t11 = T[:nk, :nk]
            # print(nk)
            if np.linalg.matrix_rank(z11) < nk:
                raise ValueError(
                    "Invertibility condition violated: z11 is singular")
            stable_count = sum(eigenvals < 1)
            if debug:
                print(
                    f"Stable eigenvalues: {stable_count}/{nk} (should be {nk})")
            if stable_count != nk:
                print("Warning: Blanchard-Kahn conditions may not be satisfied")
            z11_inv = np.linalg.inv(z11)
            P = np.real(z11 @ np.linalg.solve(s11, t11) @ z11_inv)
            F = np.real(z21 @ z11_inv)
            # Store results
            self.f = F
            self.p = P
            if debug:
                print(f"\nFinal matrices:")
                print(f"F (controls = F * states): {F.shape}")
                print(f"P (state transition): {P.shape}")
            return F, P
        # Solve using Klein's method
        F, P = solve_klein(A, B, C, self.n_states)
        return F, P
#################################

    def _compute_irfs(self, T=51, t0=1, shocks=None, center=True, normalize=True):
        """
        Compute impulse response functions (IRFs) adjusted to match the behavior of the impulse method.

        Parameters:
        - T (int): Number of periods (default: 51).
        - t0 (int): Time period when shocks are applied (default: 1).
        - shocks (dict): Dictionary of shock names and magnitudes (default: None).
        - center (bool): If True, return deviations; if False, return levels or log levels (default: True).
        - normalize (bool): If True, normalize linear approximation IRFs by steady state (default: True).

        Returns:
        - pd.DataFrame: DataFrame containing IRFs for states and controls.
        """
        if self.f is None or self.p is None:
            raise ValueError("Model matrices f and p must be defined.")

        if not self.shocks:
            raise ValueError("No shocks defined in the model.")

        # Prepare shocks
        if shocks is None:
            shocks = {shock: 0.01 for shock in self.shocks}

        n_exo_states = len([s for s in self.states if s in self.exo_states])
        ordered_vars = self.states + self.controls
        ss_values = np.array([self.steady_state[var] for var in ordered_vars])

        eps = np.zeros((T, len(self.shocks)))
        for shock, magnitude in shocks.items():
            if shock in self.shocks:
                eps[t0, self.shocks.index(shock)] = magnitude

        # Shock impact matrix
        B = np.zeros((self.n_states, len(self.shocks)))
        for i, shock in enumerate(self.shocks):
            if i < n_exo_states:
                exo_state_idx = self.states.index(self.exo_states[i])
                B[exo_state_idx, i] = 1.0

        # Simulation
        s = np.zeros((T + 1, self.n_states))
        u = np.zeros((T, self.n_controls))

        for i in range(T):
            if i == t0:
                s[i + 1] = self.p @ s[i] + B @ eps[i]
            else:
                s[i + 1] = self.p @ s[i]
            u[i] = self.f @ s[i + 1]  # Adjusted to use s[i+1] like impulse

        s = s[1:]  # States from t=1 to T
        sim = np.hstack((s, u))  # s: t=1 to T, u: t=0 to T-1

        # Variable names
        var_cols = [f"{v}_t" for v in ordered_vars]

        # Check for normalization feasibility
        if self.approximation != 'log_linear' and normalize:
            if np.any(np.isclose(ss_values, 0)):
                warnings.warn(
                    'Steady state contains zeros so normalize set to False.', stacklevel=2)
                normalize = False

        # Create DataFrame with raw simulated data
        sim_df = pd.DataFrame(sim, columns=var_cols)

        # Apply output transformations
        if not center:
            if self.approximation == 'log_linear':
                sim_df = sim_df + np.log(ss_values)  # Return log levels
            else:
                sim_df = sim_df + ss_values  # Return levels
        if normalize and self.approximation != 'log_linear':
            sim_df = sim_df / ss_values  # Normalize by steady state

        # Include shocks in the output
        shock_cols = [f"{shock}_t" for shock in self.shocks]
        eps_df = pd.DataFrame(eps, columns=shock_cols)
        irfs = pd.concat([eps_df, sim_df], axis=1)

        # Assuming var_cols is defined (e.g., from sim_df.columns)
        var_cols = sim_df.columns.tolist()

        # Replace the assignment with a dictionary creation
        self.irfs = {}
        for shock in self.shocks:
            shock_col = f"{shock}_t"
            if shock_col in irfs.columns:
                self.irfs[shock] = irfs[[shock_col] +
                                        var_cols].rename(columns={shock_col: shock})
            else:
                print(f"Warning: Shock column {shock_col} not found in irfs.")
        return self.irfs

    def plot_irfs(self, shock_names=None, T=41, scale=100, figsize=(12, 4), lw=5, alpha=0.5,
                  title_prefix="IRF", ylabel="Percentage Deviation"):
        """
        Plot impulse response functions (IRFs) with separate subplots for each exogenous state
        and other variables, for each shock.

        Args:
            shock_names (list, optional): Shocks to plot. Defaults to all shocks.
            T (int, optional): Number of periods for IRF computation. Default: 41.
            scale (float, optional): Scaling factor for IRF values. Default: 100.
            figsize (tuple, optional): Figure size. Default: (12, 4).
            lw (float, optional): Line width. Default: 5.
            alpha (float, optional): Line transparency. Default: 0.5.
            title_prefix (str, optional): Title prefix. Default: "IRF".
            ylabel (str, optional): Y-axis label. Default: "Percentage Deviation".

        Returns:
            None
        """
        # Ensure IRFs are computed
        if not hasattr(self, 'irfs') or not self.irfs:
            self._compute_irfs(T=T, t0=1, shocks=None, center=True)
        if not isinstance(self.irfs, dict) or not self.irfs:
            raise ValueError(
                "irfs must be a non-empty dictionary of Pandas DataFrames.")

        # Determine shocks to plot
        if shock_names is None:
            shock_names = list(self.irfs.keys())
        else:
            for sh in shock_names:
                if sh not in self.irfs:
                    raise ValueError(
                        f"Shock '{sh}' not found in irfs dictionary.")

        # Helper functions for column names and labels
        def get_col_name(v):
            prefix = 'hat_' if self.approximation == 'log_linear' else ''
            return f"{prefix}{v}_t"

        def get_var_from_col(col):
            if col.startswith('hat_'):
                return col[4:-2]
            else:
                return col[:-2]

        # Plot for each shock
        for sh in shock_names:
            df = self.irfs[sh]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"irfs['{sh}'] must be a Pandas DataFrame.")

            # Identify the corresponding exogenous state
            i = self.shocks.index(sh)
            if i >= len(self.exo_states):
                raise ValueError(
                    f"No corresponding exogenous state for shock '{sh}'.")
            exo_state = self.exo_states[i]
            exo_col = get_col_name(exo_state)

            # List of other variables to plot
            other_cols = [get_col_name(v)
                          for v in self.variables if v != exo_state]

            # Verify columns exist in DataFrame
            if exo_col not in df.columns:
                raise ValueError(
                    f"Column '{exo_col}' not found for shock '{sh}'.")
            missing_cols = [col for col in other_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Columns {missing_cols} not found for shock '{sh}'.")

            # Scale the IRF data
            df_scaled = df[[exo_col] + other_cols] * scale
            T_plot = len(df)

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Subplot 1: IRF of the exogenous state
            ax1.set_title(f"{title_prefix}: {exo_state} ({sh})")
            ax1.set_xlabel("Time")
            ax1.set_ylabel(ylabel)
            ax1.grid(True)
            ax1.plot(range(T_plot),
                     df_scaled[exo_col], lw=lw, alpha=alpha, label=exo_state)
            ax1.legend(loc='upper right')

            # Subplot 2: IRFs of all other variables
            ax2.set_title(f"{title_prefix}: Other Variables ({sh})")
            ax2.set_xlabel("Time")
            ax2.set_ylabel(ylabel)
            ax2.grid(True)
            for col in other_cols:
                var_name = get_var_from_col(col)
                ax2.plot(range(T_plot),
                         df_scaled[col], lw=lw, alpha=alpha, label=var_name)
            ax2.legend(loc='upper right')

            # Adjust y-axis limits for clarity
            max_y1 = df_scaled[exo_col].max() * 1.1
            min_y1 = df_scaled[exo_col].min() * 1.1
            ax1.set_ylim(min_y1 if min_y1 < 0 else 0,
                         max_y1 if max_y1 > 0 else 1)
            ax1.set_xlim(0, T_plot-1)

            max_y2 = df_scaled[other_cols].max().max() * 1.1
            min_y2 = df_scaled[other_cols].min().min() * 1.1
            ax2.set_ylim(min_y2 if min_y2 < 0 else 0,
                         max_y2 if max_y2 > 0 else 1)
            ax2.set_xlim(0, T_plot-1)

            # Finalize and display the plot
            plt.tight_layout()
            plt.show()

    def simulate(self, T=51, drop_first=300, covariance_matrix=None, seed=None, center=True, normalize=True):
        """
        Simulate the DSGE model dynamics, adjusted to match the behavior of stoch_sim.

        Parameters:
        -----------
        T : int, optional
            Number of periods to simulate (default: 51).
        drop_first : int, optional
            Number of initial periods to discard (default: 300).
        covariance_matrix : array-like, optional
            Covariance matrix for shocks (n_shocks x n_shocks).
            Defaults to diagonal matrix from shock_variance.
        seed : int, optional
            Random seed for reproducibility.
        center : bool, optional
            If True, return deviations; if False, return levels or log levels (default: True).
        normalize : bool, optional
            If True, normalize linear approximation simulations by steady state (default: True).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing simulated shocks and variables.
        """

        if self.f is None or self.p is None:
            raise ValueError("Model must be solved before simulation.")

        n_states = self.n_states
        n_costates = self.n_controls
        n_shocks = len(self.shocks)
        n_exo_states = self.n_exo_states

        # Set covariance matrix
        if covariance_matrix is None:
            variances = [0.01**2 for shock in self.shocks]
            covariance_matrix = np.diag(variances)
        else:
            covariance_matrix = np.array(covariance_matrix)
            if covariance_matrix.shape != (n_shocks, n_shocks):
                raise ValueError(
                    f"covariance_matrix must be {n_shocks}x{n_shocks}")

        # Generate shocks
        rng = np.random.default_rng(seed)
        eps = rng.multivariate_normal(
            np.zeros(n_shocks), covariance_matrix, drop_first + T)

        # Initialize state and control arrays
        # States from t=0 to t=drop_first + T
        s = np.zeros((drop_first + T + 1, n_states))
        # Controls from t=0 to t=drop_first + T - 1
        u = np.zeros((drop_first + T, n_costates))

        # Create shock impact matrix (B): maps shocks to exogenous states
        B = np.zeros((n_states, n_shocks))
        for i, shock in enumerate(self.shocks):
            if i < n_exo_states:
                exo_state_idx = self.states.index(self.exo_states[i])
                B[exo_state_idx, i] = 1.0

        # Simulate dynamics
        for t in range(drop_first + T):
            s[t + 1] = self.p @ s[t] + B @ eps[t]  # Next state
            # Control based on next state
            u[t] = self.f @ s[t + 1]

        # Compute deviations (s[t+1], u[t]) for t=drop_first to t=drop_first + T - 1
        sim_deviations = np.hstack(
            (s[drop_first + 1:drop_first + T + 1], u[drop_first:drop_first + T]))

        # Variable columns
        var_cols = [f"{v}_t" for v in self.variables]

        # Steady state values
        ss_values = np.array([self.steady_state[v] for v in self.variables])

        # Check for normalization
        if normalize and not self.approximation == 'log_linear' and np.any(np.isclose(ss_values, 0)):
            warnings.warn(
                'Steady state contains zeros so normalize set to False.', stacklevel=2)
            normalize = False

        # Transform based on center and normalize
        if center:
            sim_out = sim_deviations
        else:
            if self.approximation == "log_linear":
                sim_out = sim_deviations + np.log(ss_values)  # Log levels
            else:
                sim_out = sim_deviations + ss_values          # Levels

        if not self.approximation == "log_linear" and normalize:
            sim_out = sim_out / ss_values

        # Create DataFrame with simulated variables
        vars_ord = self._reorder_variables()
        col_names = [f"{v}_t" for v in vars_ord]
        sim_df = pd.DataFrame(sim_out, columns=col_names)

        # Include shocks
        shock_cols = [f"{sh}_t" for sh in self.shocks]
        eps_df = pd.DataFrame(eps[drop_first:], columns=shock_cols)

        # Combine shocks and variables
        self.simulated = pd.concat([eps_df, sim_df], axis=1)

        return self.simulated

    def simulations(self, title="The Model Simulation", figsize=(12, 8), save_path=None):
        """
        Plot exogenous states separately and endogenous variables together from the simulated data.

        Parameters:
        -----------
        title : str, optional
            Title of the plot (default: "Model Model Simulation").
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (12, 8)).
        save_path : str, optional
            File path to save the plot (e.g., 'plot.png'). If None, displays the plot (default: None).

        Returns:
        --------
        None
            Displays or saves the plot.
        """
        if self.simulated is None or not isinstance(self.simulated, pd.DataFrame):
            raise ValueError(
                "self.simulated must be a pandas DataFrame containing simulation results.")

        # Identify columns
        shock_cols = [f"{sh}_t" for sh in self.shocks]
        # print(self.exo_states)
        exo_state_cols = [
            f"{s}_t" for s in self.exo_states if f"{s}_t" in self.simulated.columns]
        endo_cols = [
            f"{v}_t" for v in self.variables if v not in self.exo_states and f"{v}_t" in self.simulated.columns]

        if not exo_state_cols and not endo_cols:
            raise ValueError(
                "No exogenous states or endogenous variables found in self.simulated.")

        # Number of subplots: one for each exogenous state + one for endogenous variables
        n_exo = len(exo_state_cols)
        n_subplots = n_exo + 1 if endo_cols else n_exo

        # Create figure and subplots
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
        if n_subplots == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        # Plot exogenous states
        for i, col in enumerate(exo_state_cols):
            axes[i].plot(self.simulated.index, self.simulated[col],
                         label=col, color=f"C{i}")
            axes[i].set_ylabel(col)
            axes[i].legend(loc="upper right")
            axes[i].grid(True)

        # Plot endogenous variables together
        if endo_cols:
            for col in endo_cols:
                axes[n_exo].plot(self.simulated.index,
                                 self.simulated[col], label=col)
            axes[n_exo].set_ylabel("Endogenous Variables")
            axes[n_exo].legend(loc="upper right")
            axes[n_exo].grid(True)

        # Set common labels and title
        axes[-1].set_xlabel("Time")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title

        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

############################################################################################
########################### Non Linear DSGE ################################################
############################################################################################


@dataclass
class DSGEVariable:
    """Represents a DSGE model variable with its properties."""
    name: str
    bounds: Tuple[float, float]
    steady_state: Optional[float] = None
    description: str = ""
    is_state: bool = True
    is_control: bool = False
    is_shock: bool = False


@dataclass
class DSGEParameter:
    """Represents a DSGE model parameter."""
    name: str
    value: float
    description: str = ""
    bounds: Optional[Tuple[float, float]] = None


class nonlinear_dsge:
    """
    General-purpose nonlinear DSGE model solver using projection methods.

    This class provides a flexible framework for solving DSGE models with:
    - Arbitrary state and control variables
    - User-defined equilibrium conditions
    - Multiple solution algorithms (collocation, Galerkin, least squares)
    - Comprehensive solution analysis and validation

    The class maintains full theoretical rigor while providing an intuitive interface
    for economists to specify and solve their models.
    """

    def __init__(self, name: str = "DSGE Model"):
        """Initialize the DSGE model framework."""
        self.name = name
        self.parameters: Dict[str, DSGEParameter] = {}
        self.variables: Dict[str, DSGEVariable] = {}
        self.equations: List[Callable] = []
        self.utility_function: Optional[Callable] = None
        self.production_function: Optional[Callable] = None
        self.shock_processes: Dict[str, Dict] = {}

        # Solution components
        self.solver = None
        self.coefficients = None
        self.solution_info = None
        self.is_solved = False

        # State organization
        self.state_vars: List[str] = []
        self.control_vars: List[str] = []
        self.shock_vars: List[str] = []

        print(f"Initialized {self.name}")

    def add_parameter(self, name: str, value: float, description: str = "",
                      bounds: Optional[Tuple[float, float]] = None):
        """Add a parameter to the model."""
        self.parameters[name] = DSGEParameter(name, value, description, bounds)
        # Make parameter accessible as attribute for convenience
        setattr(self, name, value)
        return self

    def add_variable(self, name: str, bounds: Tuple[float, float],
                     steady_state: Optional[float] = None,
                     description: str = "", variable_type: str = "state"):
        """Add a variable to the model."""
        is_state = variable_type in ["state", "both"]
        is_control = variable_type in ["control", "both"]
        is_shock = variable_type == "shock"

        self.variables[name] = DSGEVariable(
            name, bounds, steady_state, description, is_state, is_control, is_shock
        )

        # Organize variables by type
        if is_state:
            self.state_vars.append(name)
        if is_control:
            self.control_vars.append(name)
        if is_shock:
            self.shock_vars.append(name)

        return self

    def set_utility_function(self, utility_func: Callable, marginal_utility_func: Optional[Callable] = None):
        """Set the utility function and optionally its derivative."""
        self.utility_function = utility_func
        if marginal_utility_func:
            self.marginal_utility_function = marginal_utility_func
        else:
            # Try to compute numerical derivative if not provided
            self.marginal_utility_function = self._numerical_derivative(
                utility_func)
        return self

    def set_production_function(self, production_func: Callable):
        """Set the production function."""
        self.production_function = production_func
        return self

    def add_shock_process(self, shock_name: str, persistence: float,
                          volatility: float, process_type: str = "AR1"):
        """Add a stochastic shock process."""
        self.shock_processes[shock_name] = {
            'persistence': persistence,
            'volatility': volatility,
            'type': process_type
        }
        return self

    def add_equilibrium_condition(self, equation_func: Callable):
        """Add an equilibrium condition (e.g., Euler equation, FOC)."""
        self.equations.append(equation_func)
        return self

    def compute_steady_state(self, method: str = "analytical"):
        """Compute model steady state."""
        if method == "analytical":
            # User should provide analytical steady state values
            for var_name, var in self.variables.items():
                if var.steady_state is None:
                    warnings.warn(f"No steady state provided for {var_name}")
        elif method == "numerical":
            # Implement numerical steady state solver
            self._solve_steady_state_numerically()

        # Update variable steady states and print summary
        print("\n=== STEADY STATE ===")
        for var_name, var in self.variables.items():
            if var.steady_state is not None:
                print(f"  {var_name}: {var.steady_state:.6f}")

        return self

    def setup_solver(self, approximation_orders: Dict[str, int]):
        """Set up the projection solver with specified polynomial orders."""
        # Organize variables in consistent order
        all_vars = self.state_vars + self.shock_vars

        order_vector = [approximation_orders.get(var, 5) for var in all_vars]
        lower_bounds = [self.variables[var].bounds[0] for var in all_vars]
        upper_bounds = [self.variables[var].bounds[1] for var in all_vars]

        # Import and initialize the ProjectionSolver
        self.solver = ProjectionSolver(
            order_vector, lower_bounds, upper_bounds)

        print(f"\nSolver initialized:")
        print(f"  Variables: {all_vars}")
        print(f"  Polynomial orders: {dict(zip(all_vars, order_vector))}")
        print(f"  Basis functions: {self.solver.basis_size}")

        return self

    def _create_residual_function(self):
        """Create the residual function for the solver."""
        def residual_function(grid, policy_values, coeffs):
            """
            Evaluate equilibrium condition residuals at grid points.

            Parameters:
            -----------
            grid : ndarray
                State space grid points (n_points x n_states)
            policy_values : ndarray  
                Policy function values at grid points
            coeffs : ndarray
                Current coefficient guess

            Returns:
            --------
            residuals : ndarray
                Residual values at each grid point
            """
            n_points = grid.shape[0]
            residuals = np.zeros(n_points)

            # Get shock process parameters for expectation calculation
            shock_info = list(self.shock_processes.values())[
                0] if self.shock_processes else None

            for i in range(n_points):
                # Current state values
                current_state = self._grid_to_state_dict(grid[i])

                # Policy function gives us control variables
                if len(self.control_vars) == 1:
                    current_state[self.control_vars[0]] = policy_values[i]
                else:
                    for j, control_var in enumerate(self.control_vars):
                        current_state[control_var] = policy_values[i,
                                                                   j] if policy_values.ndim > 1 else policy_values[i]

                # Evaluate equilibrium conditions
                residual_sum = 0.0

                for eq_func in self.equations:
                    try:
                        # If equation involves expectations, compute them
                        if self._equation_has_expectations(eq_func):
                            residual_sum += self._evaluate_equation_with_expectations(
                                eq_func, current_state, grid[i], coeffs, shock_info
                            )
                        else:
                            residual_sum += eq_func(current_state,
                                                    self.parameters)
                    except Exception as e:
                        residuals[i] = 1e6  # Penalty for evaluation errors
                        break
                else:
                    residuals[i] = residual_sum

            return residuals

        return residual_function

    def _grid_to_state_dict(self, grid_point):
        """Convert grid point to state dictionary."""
        state_dict = {}
        all_vars = self.state_vars + self.shock_vars
        for i, var_name in enumerate(all_vars):
            state_dict[var_name] = grid_point[i]
        return state_dict

    def _equation_has_expectations(self, equation_func: Callable) -> bool:
        """Check if equation involves expectations (heuristic based on signature)."""
        sig = inspect.signature(equation_func)
        # More than (state, params) suggests expectations
        return len(sig.parameters) > 2

    def _evaluate_equation_with_expectations(self, eq_func, current_state, grid_point, coeffs, shock_info):
        """Evaluate equation involving expectations using quadrature."""
        if shock_info is None:
            return eq_func(current_state, self.parameters)

        # Gauss-Hermite quadrature for expectations
        n_nodes = 5
        nodes, weights = self.solver.cheb_basis.gauss_hermite_nodes(
            n_nodes, shock_info['volatility'])

        expectation = 0.0
        for node, weight in zip(nodes, weights):
            # Compute next period shock
            current_shock = current_state.get(
                self.shock_vars[0], 1.0) if self.shock_vars else 1.0
            next_shock = np.exp(shock_info['persistence'] * np.log(current_shock) +
                                shock_info['volatility'] * np.log(node))

            # Create next period state
            next_state = current_state.copy()
            if self.shock_vars:
                next_state[self.shock_vars[0]] = next_shock

            # Evaluate equation for this realization
            try:
                value = eq_func(current_state, self.parameters,
                                next_state, self.solver, coeffs)
                expectation += weight * value
            except:
                expectation += weight * 1e6  # Penalty for errors

        return expectation

    def solve(self, method: str = "collocation",
              initial_policy: Optional[Callable] = None,
              solver_options: Optional[Dict] = None,
              verbose: bool = True) -> 'nonlinear_dsge':
        """
        Solve the DSGE model using specified projection method.

        Parameters:
        -----------
        method : str
            Solution method: 'collocation', 'galerkin', or 'least_squares'
        initial_policy : callable, optional
            Function to generate initial policy guess
        solver_options : dict, optional
            Additional options for the solver
        verbose : bool
            Print solution progress

        Returns:
        --------
        self : NonlinearDSGE
            Returns self for method chaining
        """
        if self.solver is None:
            raise ValueError("Must call setup_solver() before solve()")

        if not self.equations:
            raise ValueError("No equilibrium conditions specified")

        # Set default solver options
        default_options = {'maxit': 5000, 'stopc': 1e-8}
        options = {**default_options, **(solver_options or {})}

        # Create initial guess
        if initial_policy is None:
            initial_guess = self._default_initial_guess()
        else:
            initial_guess = self._create_initial_guess(initial_policy)

        # Create residual function
        residual_func = self._create_residual_function()

        print(f"\nSolving {self.name} using {method} method...")

        # Solve using specified method
        if method == "collocation":
            coeffs, crit = self.solver.solve_collocation(
                residual_func, initial_guess, verbose=verbose, **options
            )
        elif method == "galerkin":
            coeffs, crit = self.solver.solve_galerkin(
                residual_func, initial_guess, verbose=verbose, **options
            )
        elif method == "least_squares":
            coeffs, crit = self.solver.solve_least_squares(
                residual_func, initial_guess, verbose=verbose, **options
            )
        else:
            raise ValueError(f"Unknown solution method: {method}")

        # Store solution
        self.coefficients = coeffs
        self.solution_info = {
            'method': method,
            'convergence': crit,
            'converged': crit[1] < options['stopc']
        }
        self.is_solved = True

        if verbose:
            print(f"\nSolution completed!")
            print(f"  Converged: {self.solution_info['converged']}")
            print(f"  Final criterion: {crit[1]:.2e}")
            print(f"  Iterations: {int(crit[4])}")

        return self

    def _default_initial_guess(self):
        """Create default initial guess based on steady states."""
        nodes = self.solver.cheb_basis.funnode()
        grid = self.solver.cheb_basis.gridmake(nodes)

        # Simple constant policy at steady state values
        if self.control_vars:
            control_ss = self.variables[self.control_vars[0]].steady_state
            if control_ss is None:
                control_ss = np.mean(
                    [self.variables[self.control_vars[0]].bounds])
            initial_values = np.full(grid.shape[0], control_ss)
        else:
            initial_values = np.zeros(grid.shape[0])

        # Fit to basis
        basis_matrix = self.solver.cheb_basis.funbas(grid)
        coeffs = np.linalg.lstsq(basis_matrix, initial_values, rcond=None)[0]
        return coeffs

    def _create_initial_guess(self, policy_func: Callable):
        """Create initial guess from user-provided policy function."""
        nodes = self.solver.cheb_basis.funnode()
        grid = self.solver.cheb_basis.gridmake(nodes)

        initial_values = np.zeros(grid.shape[0])
        for i, grid_point in enumerate(grid):
            state_dict = self._grid_to_state_dict(grid_point)
            initial_values[i] = policy_func(state_dict, self.parameters)

        # Fit to basis
        basis_matrix = self.solver.cheb_basis.funbas(grid)
        coeffs = np.linalg.lstsq(basis_matrix, initial_values, rcond=None)[0]
        return coeffs

    def evaluate_policy(self, state_points: np.ndarray) -> np.ndarray:
        """
        Evaluate the solved policy function at given state points.

        Parameters:
        -----------
        state_points : ndarray
            State space points where to evaluate policy (n_points x n_states)

        Returns:
        --------
        policy_values : ndarray
            Policy function values at the points
        """
        if not self.is_solved:
            raise ValueError("Model must be solved before evaluating policy")

        return self.solver.evaluate_solution(self.coefficients, state_points)

    def validate_solution(self, n_test_points: int = 100,
                          random_seed: Optional[int] = None) -> Dict[str, float]:
        """
        Validate the solution by checking equilibrium conditions.

        Parameters:
        -----------
        n_test_points : int
            Number of random test points
        random_seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        validation_metrics : dict
            Dictionary containing various error metrics
        """
        if not self.is_solved:
            raise ValueError("Model must be solved before validation")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random test points
        test_points = np.zeros(
            (n_test_points, len(self.state_vars) + len(self.shock_vars)))
        all_vars = self.state_vars + self.shock_vars

        for i, var_name in enumerate(all_vars):
            bounds = self.variables[var_name].bounds
            test_points[:, i] = np.random.uniform(
                bounds[0], bounds[1], n_test_points)

        # Evaluate policy and residuals
        test_policy = self.evaluate_policy(test_points)
        residual_func = self._create_residual_function()
        test_residuals = residual_func(
            test_points, test_policy, self.coefficients)

        # Compute validation metrics
        metrics = {
            'mean_abs_error': np.mean(np.abs(test_residuals)),
            'max_abs_error': np.max(np.abs(test_residuals)),
            'rms_error': np.sqrt(np.mean(test_residuals**2)),
            'l_infinity_norm': np.max(np.abs(test_residuals)),
            'n_test_points': n_test_points
        }

        print(f"\n=== SOLUTION VALIDATION ===")
        print(
            f"Equilibrium condition errors at {n_test_points} random points:")
        print(f"  Mean absolute error: {metrics['mean_abs_error']:.2e}")
        print(f"  Maximum absolute error: {metrics['max_abs_error']:.2e}")
        print(f"  RMS error: {metrics['rms_error']:.2e}")
        print(f"  L∞ norm: {metrics['l_infinity_norm']:.2e}")

        return metrics

    def analyze_policy(self, n_plot_points: int = 50,
                       shock_values: Optional[List[float]] = None,
                       figsize: Tuple[int, int] = (15, 10)):
        """
        Analyze and visualize the solved policy functions.

        Parameters:
        -----------
        n_plot_points : int
            Number of points for plotting
        shock_values : list, optional
            Specific shock values to analyze
        figsize : tuple
            Figure size for plots
        """
        if not self.is_solved:
            raise ValueError("Model must be solved before analysis")

        if len(self.state_vars) == 0:
            print("No state variables to plot")
            return

        # Default shock values if not provided
        if shock_values is None and self.shock_vars:
            shock_var = self.shock_vars[0]
            bounds = self.variables[shock_var].bounds
            shock_values = [bounds[0], (bounds[0] + bounds[1])/2, bounds[1]]
        elif not self.shock_vars:
            shock_values = [1.0]  # No shocks case

        # Create evaluation grid for first state variable
        state_var = self.state_vars[0]
        state_bounds = self.variables[state_var].bounds
        state_eval = np.linspace(
            state_bounds[0], state_bounds[1], n_plot_points)

        # Set up plots
        n_plots = 2 if len(self.control_vars) > 0 else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        # Plot policy functions
        for shock_val in shock_values:
            # Create evaluation grid
            if len(self.shock_vars) > 0:
                grid_eval = np.column_stack([state_eval,
                                             np.full(n_plot_points, shock_val)])
            else:
                grid_eval = state_eval.reshape(-1, 1)

            # Evaluate policy
            policy_values = self.evaluate_policy(grid_eval)

            # Plot policy function
            axes[0].plot(state_eval, policy_values,
                         label=f'{self.shock_vars[0] if self.shock_vars else "No shock"}={shock_val:.2f}')

        axes[0].set_xlabel(f'{state_var}')
        axes[0].set_ylabel(
            f'{self.control_vars[0] if self.control_vars else "Policy"}')
        axes[0].set_title('Policy Function')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot consumption/savings rate if applicable
        if n_plots > 1 and self.production_function:
            for shock_val in shock_values:
                if len(self.shock_vars) > 0:
                    grid_eval = np.column_stack([state_eval,
                                                 np.full(n_plot_points, shock_val)])
                else:
                    grid_eval = state_eval.reshape(-1, 1)

                policy_values = self.evaluate_policy(grid_eval)

                # Compute implied consumption or savings rate
                production_values = np.array([
                    self.production_function(
                        self._grid_to_state_dict(point), self.parameters)
                    for point in grid_eval
                ])

                savings_rate = policy_values / production_values
                axes[1].plot(state_eval, savings_rate,
                             label=f'{self.shock_vars[0] if self.shock_vars else "No shock"}={shock_val:.2f}')

            axes[1].set_xlabel(f'{state_var}')
            axes[1].set_ylabel('Savings Rate')
            axes[1].set_title('Optimal Savings Rate')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return self

    def simulate(self, T: int, initial_state: Optional[Dict] = None,
                 random_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate the model for T periods.

        Parameters:
        -----------
        T : int
            Number of periods to simulate
        initial_state : dict, optional
            Initial state values (default: steady state)
        random_seed : int, optional
            Random seed for shock simulation

        Returns:
        --------
        simulation : dict
            Dictionary with time series for all variables
        """
        if not self.is_solved:
            raise ValueError("Model must be solved before simulation")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize simulation arrays
        simulation = {}
        all_vars = self.state_vars + self.control_vars + self.shock_vars
        for var in all_vars:
            simulation[var] = np.zeros(T)

        # Set initial conditions
        if initial_state is None:
            for var in all_vars:
                if self.variables[var].steady_state is not None:
                    simulation[var][0] = self.variables[var].steady_state
                else:
                    bounds = self.variables[var].bounds
                    simulation[var][0] = (bounds[0] + bounds[1]) / 2
        else:
            for var, value in initial_state.items():
                simulation[var][0] = value

        # Simulate forward
        for t in range(T-1):
            # Current state point
            current_point = np.array(
                [[simulation[var][t] for var in self.state_vars + self.shock_vars]])

            # Evaluate policy
            policy_value = self.evaluate_policy(current_point)[0]
            if self.control_vars:
                simulation[self.control_vars[0]][t] = policy_value

            # Update state variables according to transition equations
            self._update_state_variables(simulation, t)

            # Update shock processes
            self._update_shocks(simulation, t)

        # Final period policy
        if t == T-2:  # Fill in last period
            final_point = np.array(
                [[simulation[var][T-1] for var in self.state_vars + self.shock_vars]])
            policy_value = self.evaluate_policy(final_point)[0]
            if self.control_vars:
                simulation[self.control_vars[0]][T-1] = policy_value

        return simulation

    def _update_state_variables(self, simulation: Dict, t: int):
        """Update state variables based on model dynamics."""
        # This is model-specific and should be overridden or specified by user
        # For now, implement a generic capital accumulation
        if 'k' in self.state_vars and 'k' in self.control_vars:
            simulation['k'][t+1] = simulation['k'][t]  # Identity for now

    def _update_shocks(self, simulation: Dict, t: int):
        """Update shock processes."""
        for shock_var in self.shock_vars:
            if shock_var in self.shock_processes:
                process = self.shock_processes[shock_var]
                ρ = process['persistence']
                σ = process['volatility']
                ε = np.random.normal(0, 1)

                current_shock = simulation[shock_var][t]
                simulation[shock_var][t +
                                      1] = np.exp(ρ * np.log(current_shock) + σ * ε)

    def summary(self):
        """Print a comprehensive summary of the model."""
        print(f"\n{'='*50}")
        print(f"{self.name.upper()} - MODEL SUMMARY")
        print(f"{'='*50}")

        print(f"\nPARAMETERS ({len(self.parameters)}):")
        for param in self.parameters.values():
            bounds_str = f" ∈ {param.bounds}" if param.bounds else ""
            print(f"  {param.name} = {param.value:.4f}{bounds_str}")
            if param.description:
                print(f"    {param.description}")

        print(f"\nVARIABLES ({len(self.variables)}):")
        for var in self.variables.values():
            type_str = []
            if var.is_state:
                type_str.append("state")
            if var.is_control:
                type_str.append("control")
            if var.is_shock:
                type_str.append("shock")

            ss_str = f", ss={var.steady_state:.4f}" if var.steady_state else ""
            print(f"  {var.name} ∈ {var.bounds} ({'/'.join(type_str)}{ss_str})")
            if var.description:
                print(f"    {var.description}")

        print(f"\nEQUILIBRIUM CONDITIONS ({len(self.equations)}):")
        for i, eq in enumerate(self.equations):
            print(
                f"  {i+1}. {eq.__name__ if hasattr(eq, '__name__') else 'Equation'}")

        if self.is_solved:
            print(f"\nSOLUTION STATUS:")
            print(f"  Method: {self.solution_info['method']}")
            print(f"  Converged: {self.solution_info['converged']}")
            print(f"  Final error: {self.solution_info['convergence'][1]:.2e}")
        else:
            print(f"\nSOLUTION STATUS: Not solved")

    def _numerical_derivative(self, func: Callable, h: float = 1e-8):
        """Create numerical derivative function."""
        def derivative(x):
            return (func(x + h) - func(x - h)) / (2 * h)
        return derivative

    # def export_solution(self, filename: str):
    #     """Export solution coefficients and model info to file."""
    #     if not self.is_solved:
    #         raise ValueError("Model must be solved before export")

    #     export_data = {
    #         'model_name': self.name,
    #         'parameters': {name: param.value for name, param in self.parameters.items()},
    #         'variables': {name: {'bounds': var.bounds, 'steady_state': var.steady_state}
    #                      for name, var in self.variables.items()},
    #         'coefficients': self.coefficients.tolist(),
    #         'solution_info': self.solution_info
    #     }

    #     import json
    #     with open(filename, 'w') as f:
    #         json.dump(export_data, f, indent=2)

    #     print(f"Solution exported to {filename}")

    # def load_solution(self, filename: str):
    #     """Load solution from file."""
    #     import json
    #     with open(filename, 'r') as f:
    #         data = json.load(f)

    #     self.coefficients = np.array(data['coefficients'])
    #     self.solution_info = data['solution_info']
    #     self.is_solved = True

    #     print(f"Solution loaded from {filename}")
    #     return self
