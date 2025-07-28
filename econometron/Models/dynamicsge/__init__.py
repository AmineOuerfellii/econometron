# # ###########################
# Beginning with imports
from sympy import symbols, Symbol, Matrix , collect , S, exp ,log
import re
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import ordqz,qz,schur
import matplotlib.animation as animation
import warnings

####################
__all__ = ['linear_dsge', 'nonlinear_dsge']
####
# Time symbol
t = Symbol('t', integer=True)

# Transformations for parsing equations
_transformations = standard_transformations + (implicit_multiplication_application,)

#################

class linear_dsge():
    
  def __init__(self,equations=None,variables=None,exo_states=None,endo_states=None,parameters=None,approximation=None,normalize=None,shocks=None):
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
    self.endo_states=endo_states if endo_states is not None else []
    self.exo_states=exo_states if exo_states is not None else []
    self.states=self.exo_states+self.endo_states
    self.controls=[var for var in self.variables if var not in self.states]
    ######################
    self.n_vars=len(self.variables)
    self.n_equations=len(self.equations_list)
    self.n_states = len(self.states)
    self.n_exo_states = len(exo_states) if exo_states else None
    self.n_endo_states = len(endo_states) if endo_states else None
    self.n_controls=len(self.controls)
    #####################
    self.steady_state=None
    self.f=None
    self.p=None
    self.c=None
    ###################
    self.stoch_simulated=None
    ###################
    self.irfs = None
    ###################
    #self.shock_variances = shock_variances or {shock: 0.01**2 for shock in shock_names}
    ###################
    self.approximated=None
    self.solved=None
    #########################################################################################
  def validate_entries(self):
    # Check equations
    if not self.equations_list or not isinstance(self.equations_list, list):
        raise ValueError("Equations must be provided as a non-empty list.")
    for eq in self.equations_list:
        if not isinstance(eq, str):
            raise ValueError(f"Each equation must be a string. Invalid entry: {eq}")
        # Check for valid equation format
        if '=' not in eq:
            raise ValueError(f"Invalid equation format: {eq}. Each equation must contain an '=' sign.")
    if self.equations_list ==[]:
        raise ValueError("Equations must be provided as a non-empty list.")      
    # Check variables
    if not self.variables or not isinstance(self.variables, list):
        raise ValueError("Variables must be provided as a non-empty list.")
    # Check parameters
    if not self.parameters or not isinstance(self.parameters, dict):
        raise ValueError("Parameters must be provided as a non-empty dictionary.")
    # Check shocks
    if self.shocks is None:
        warnings.warn("No shocks specified. Model may be deterministic.", stacklevel=2)
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
        raise ValueError("Duplicate variable names found in variables list.")
    # Check for missing variables in equations
    missing_vars = [var for var in self.variables if not any(var in eq for eq in self.equations_list)]
    if missing_vars:
        warnings.warn(f"Variables {missing_vars} do not appear in any equation.", stacklevel=2)
    # Check approximation
    if self.approximation not in ['linear', 'log_linear']:
        warnings.warn(f"Unknown approximation type: {self.approximation}", stacklevel=2)
    # Check normalize
    if self.normalize and not isinstance(self.normalize, dict):
        warnings.warn("Normalize should be a dictionary mapping variable names to values.", stacklevel=2)
    return True

  def set_initial_guess(self, initial_guess):
    """
    this function sets the initial guess for the model
    """
    if not isinstance(initial_guess, list):
        raise ValueError("Initial guess must be a list.")
    if len(initial_guess) != len([var for var in self.variables if var not in self.exo_states]):
        raise ValueError("Initial guess must match the number of variables.")
    self.initial_guess = np.array(initial_guess)

  def _parse_equations(self,eq):
    """
    Remodel._parse_equations(eq), is the function resposable of parsing the equations of a certain dynmic model
    This pacakge enables Users to write function as they are in the model , no traditional techniques are used
    so a user can write his model a a "String" and this function is the parser

    parameters:

    eq: equations

    """
    local_dict={}
    for var in self.variables:
      local_dict[f"{var}_t"] = Symbol(f"{var}_t")
      local_dict[f"{var}_tp1"] = Symbol(f"{var}_tp1")
      local_dict[f"{var}_tm1"] = Symbol(f"{var}_tm1")
    for shock in self.shocks:
      local_dict[f"{shock}"] = Symbol(f"{shock}")
      local_dict[f"{shock}_t"] = Symbol(f"{shock}_t")
    for param in self.parameters:
      local_dict[param] = Symbol(param)

    eq_normalized= re.sub(r"[{\(]t([+-]1)?[}\)]", lambda m: {None: "t","+1": "tp1","-1": "tm1"}[m.group(1)], eq)
    if '=' in eq_normalized:
      left, right = eq_normalized.split('=')
      left_expr = parse_expr(left, local_dict=local_dict, transformations='all')
      right_expr = parse_expr(right, local_dict=local_dict, transformations='all')
      expr = left_expr - right_expr
    else:
      expr = parse_expr(eq_normalized, local_dict=local_dict, transformations='all')

    tp1_terms = S.Zero
    t_terms = S.Zero
    shock_terms = S.Zero
    all_vars = set(local_dict.keys())
    expr = expr.expand()
    for term in expr.as_ordered_terms():
      term_str = str(term)
      term_symbols = term.free_symbols
      is_constant = term_symbols and all(str(sym) in self.parameters for sym in term_symbols)
      has_shock = any(f"{shock}_t" in term_str for shock in self.shocks)
      has_tp1 = 'tp1' in term_str
      has_t = any(f"{var}_t" in term_str for var in self.variables) and not has_tp1
      if has_tp1 and not has_t and not has_shock:
          tp1_terms += term
      elif has_t and not has_tp1 and not has_shock:
          t_terms += term
      elif is_constant or has_shock:
          shock_terms += term
      else:
          coeff_dict = collect(term, [local_dict[f"{var}_tp1"] for var in self.variables] +
                              [local_dict[f"{var}_t"] for var in self.variables] +
                              ([local_dict[f"{shock}"] for shock in self.shocks] or [local_dict[f"{shock}_t"]for shock in self.shocks] ), evaluate=False)

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
            raise ValueError(f"Cannot convert residual to float for equation '{eq}': {residual}")

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
                  exo_ss_values[var] = 1.0 if self.approximation=="log_linear" else 0.0

      if guess is None:
          print("No initial guess provided. Using ones as default.")
          guess = np.ones(n_endogenous_vars)  # Better default than zeros
          print(guess)
      else:
          guess = np.array(guess, dtype=float)

          if len(guess) != n_endogenous_vars:
              raise ValueError(f"Initial guess must have length {n_endogenous_vars}.")

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
              raise ValueError(f"Expected {self.n_equations} residuals, got {len(residuals)}.")
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

      steady_state_dict = {var: val for var, val in zip(endogenous_vars, steady_state)}
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
        raise ValueError(f"Reordered variables do not match original set: "f"reordered={reordered}, original={self.variables}")
    return reordered

  def _Analytical_jacobians(self,debug=False):
    """
    Compute Jacobians A, B, and C for the DSGE model A E_t[y_{t+1}] = B y_t + C epsilon_t.
    Variables are ordered as [states, control], shocks follow self.shock_names order.
    Rows of A, B, C are reordered to match variable indices in ordered_vars.
    Returns:
        A (ndarray): Jacobian with respect to y_{t+1}.
        B (ndarray): Jacobian with respect to y_t.
        C (ndarray): Jacobian with respect to shocks epsilon_t.
    """
    ordered_vars = self._reorder_variables()
    if debug==True:
      print("Reordered variables:", ordered_vars)
    vars_t = [Symbol(f"{var}_t") for var in ordered_vars]
    vars_tp1 = [Symbol(f"{var}_tp1") for var in ordered_vars]
    vars_tm1 = [Symbol(f"{var}_tm1") for var in ordered_vars]

    ##Check for steady states

    if self.steady_state is None:
      raise ValueError("Steady state not computed. Call compute_ss() first")

    #Intialization of Matrices
    A = np.zeros((len(self.equations_list), len(ordered_vars)))
    B = np.zeros((len(self.equations_list), len(ordered_vars)))
    shocks = self.shocks if self.shocks else []
    C = np.zeros((len(self.equations_list), len(shocks)))
    if not shocks:
      print("Warning: No shocks identified. Check self.shock_names:", self.shocks)
    # Steady state substitution dictionary
    subs = {Symbol(f"{var}_t"): self.steady_state[var] for var in ordered_vars}
    subs.update({Symbol(f"{var}_tp1"): self.steady_state[var] for var in ordered_vars})
    subs.update({Symbol(f"{shock}"): 0.0 for shock in shocks})
    subs.update({Symbol(k): float(v) for k, v in self.parameters.items()})


    ######
    # Mapping of equations to variables
    eq_to_var = {}
    remaining_eqs = list(range(len(self.equations_list)))
    for i, eq in enumerate(self.equations_list):
        tp1_terms, _, _ = self._parse_equations(eq)
        # Look for tp1 variables to identify state transitions
        for var in ordered_vars:
            if Symbol(f"{var}_tp1") in tp1_terms.free_symbols:
                if var not in eq_to_var.values() and i in remaining_eqs:
                    eq_to_var[i] = var
                    remaining_eqs.remove(i)
                    break
        # For non-state equations, look for t variables
        if i not in eq_to_var:
            _, t_terms, _ = self._parse_equations(eq)
            for var in ordered_vars:
                if Symbol(f"{var}_t") in t_terms.free_symbols:
                    if var not in eq_to_var.values() and i in remaining_eqs:
                        eq_to_var[i] = var
                        remaining_eqs.remove(i)
                        break
    # Assign remaining equations to unassigned variables
    remaining_vars = [v for v in ordered_vars if v not in eq_to_var.values()]
    for i, eq_idx in enumerate(remaining_eqs):
        if i < len(remaining_vars):
            eq_to_var[eq_idx] = remaining_vars[i]
    if debug:
        print(self.approximation)
    # Create reordering index
    reorder_idx = [0] * len(self.equations_list)
    for eq_idx, var in eq_to_var.items():
        var_idx = ordered_vars.index(var)
        reorder_idx[eq_idx] = var_idx
    if debug==True:
      print("Equation to variable mapping:", eq_to_var)
      print("Reordering index:", reorder_idx)
    ## lineariztion in levels : approximtion =linear
    if self.approximation == 'linear':
      for i, eq in enumerate(self.equations_list):
        tp1_terms, t_terms, shock_terms = self._parse_equations(eq)
        if debug==True:
          print(f"Equation {i+1}: {eq}, shock_terms: {shock_terms}")
        for j, var in enumerate(vars_tp1):
          coeff = tp1_terms.diff(var) if tp1_terms != S.Zero else S.Zero
          A[i, j] = float(coeff.subs(subs)) if coeff != S.Zero else 0.0
        for j, var in enumerate(vars_t):
          coeff = t_terms.diff(var) if t_terms != S.Zero else S.Zero
          B[i, j] = float(coeff.subs(subs)) if coeff != S.Zero else 0.0
        for j, shock in enumerate(shocks):
          coeff = shock_terms.diff(Symbol(shock)) if shock_terms != S.Zero else S.Zero
          if debug:
            print(coeff)
            print(f"Equation {i+1}, shock {shocks[j]}, derivative: {coeff}")
          C[i, j] = float(coeff.subs(subs)) if coeff != S.Zero else 0.0
    else :
      if np.any(np.isclose([self.steady_state[var] for var in ordered_vars], 0)):
          raise ValueError("Steady state contains zeros; cannot compute log-linear Jacobians.")

      log_vars_t = [Symbol(f"log_{var}_t") for var in ordered_vars]
      log_vars_tp1 = [Symbol(f"log_{var}_tp1") for var in ordered_vars]
      log_shocks = [Symbol(f"log_{shock}") for shock in shocks]

      eqs = []
      for eq in self.equations_list:
          tp1_terms, t_terms, shock_terms = self._parse_equations(eq)
          expr = tp1_terms - t_terms - shock_terms
          if debug:
            print(f"Equation: {eq}, shock_terms: {shock_terms}")
          
          # Handle the AR(1) process equation differently (equation with log terms)
          if "log(" in str(expr):
              # This is already a log-deviation equation
              # Replace log(A_t) with log_A_t, log(A_tp1) with log_A_tp1, etc.
              subs_log = {}
              for var, log_var, log_var_tp1 in zip(ordered_vars, log_vars_t, log_vars_tp1):
                  subs_log[log(Symbol(f"{var}_t"))] = log_var
                  subs_log[log(Symbol(f"{var}_tp1"))] = log_var_tp1
              
              # Shocks are already deviations in log equations
              for shock, log_shock in zip(shocks, log_shocks):
                  subs_log[Symbol(shock)] = log_shock
                  
              expr = expr.subs(subs_log)
          else:
              # This is a level equation - needs log-linearization
              subs_ss = {}
              for var in ordered_vars:
                  subs_ss[Symbol(f"{var}_t")] = self.steady_state[var]
                  subs_ss[Symbol(f"{var}_tp1")] = self.steady_state[var]
              for shock in shocks:
                  subs_ss[Symbol(shock)] = 0
                  
              # Take derivatives for linearization
              linear_expr = expr.subs(subs_ss)  # Constant term (should be 0 at steady state)
              
              # Add linear terms
              for var, log_var in zip(ordered_vars, log_vars_t):
                  deriv = expr.diff(Symbol(f"{var}_t")).subs(subs_ss)
                  linear_expr += deriv * self.steady_state[var] * log_var
                  
              for var, log_var_tp1 in zip(ordered_vars, log_vars_tp1):
                  deriv = expr.diff(Symbol(f"{var}_tp1")).subs(subs_ss)
                  linear_expr += deriv * self.steady_state[var] * log_var_tp1
                  
              for shock, log_shock in zip(shocks, log_shocks):
                  deriv = expr.diff(Symbol(shock)).subs(subs_ss)
                  linear_expr += deriv * log_shock
                  
              expr = linear_expr
          
          # Normalize if specified
          normalize_var = getattr(self, 'normalize', None)
          if normalize_var and normalize_var in ordered_vars and not np.isclose(self.steady_state[normalize_var], 0):
              expr = expr / self.steady_state[normalize_var]
          
          eqs.append(expr)

      # Compute Jacobians
      A_mat = Matrix(eqs).jacobian(log_vars_tp1)
      B_mat = Matrix(eqs).jacobian(log_vars_t)
      C_mat = Matrix(eqs).jacobian(log_shocks)

      # Set log deviations to zero for steady state evaluation
      log_subs = {Symbol(f"log_{var}_t"): 0 for var in ordered_vars}
      log_subs.update({Symbol(f"log_{var}_tp1"): 0 for var in ordered_vars})
      log_subs.update({Symbol(f"log_{shock}"): 0 for shock in shocks})
      log_subs.update({Symbol(k): float(v) for k, v in self.parameters.items()})
      if debug:
        print(log_subs)

      A = np.array(A_mat.subs(log_subs), dtype=float)
      B = np.array(B_mat.subs(log_subs), dtype=float)
      C = np.array(C_mat.subs(log_subs), dtype=float)
      # Reorder rows
      A = A[reorder_idx, :]
      B = B[reorder_idx, :]
      C = C[reorder_idx, :]
      if debug==True:
        print("Analytical Jacobian A (rows: variables, cols: [states, control]):\n", A_mat)
        print("Analytical Jacobian B (rows: variables, cols: [states, control]):\n", B_mat)
        print("Analytical Jacobian C (rows: variables, cols: shocks ", shocks, "):\n", C_mat)
        if np.allclose(C, 0) and shocks:
            print("Warning: C matrix is all zeros. Check shock specifications or _parse_equation method.")

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
      using finite differences. Handles shocks flexibly, avoiding parameter-shock collisions.
      Returns:
          A_num (ndarray): Numerical Jacobian with respect to y_{t+1}.
          B_num (ndarray): Numerical Jacobian with respect to y_t.
          C_num (ndarray): Numerical Jacobian with respect to shocks epsilon_t.
      """
      e_s = np.array([self.steady_state[var] for var in self.variables], dtype=np.float64)
      A_num = np.zeros((len(self.equations_list), len(self.variables)))
      B_num = np.zeros((len(self.equations_list), len(self.variables)))
      
      # Use shock names consistently
      shock_names = self.shocks
      C_num = np.zeros((len(self.equations_list), len(shock_names)))
      
      if not shock_names:
          print("Warning: No shocks identified. Check self.shocks:", self.shocks)
          return A_num, B_num, C_num
      
      # Map shock names to symbols
      shock_symbols = {shock: Symbol(shock) for shock in shock_names}
      
      # Exclude shock names from parameters to avoid collision
      parameters = {k: v for k, v in self.parameters.items() if k not in shock_names}
      
      if self.approximation == 'log_linear':
          if np.any(np.isclose(e_s, 0)):
              raise ValueError("Steady state contains zeros; cannot compute log-linear Jacobians.")
          
          def psi(log_vars_fwd, log_vars_cur, log_shocks=None):
              vars_fwd = np.exp(log_vars_fwd) 
              vars_cur = np.exp(log_vars_cur) 
              shocks = log_shocks if log_shocks is not None else np.zeros(len(shock_names))
              residuals = np.zeros(len(self.equations_list))
              
              for i, eq in enumerate(self.equations_list):
                  tp1_terms, t_terms, shock_terms = self._parse_equations(eq)
                  expr = tp1_terms + t_terms + shock_terms
                  
                  if debug:
                      print(f"Equation {i+1}: {eq}, shock_terms: {shock_terms}")
                  
                  # Check if equation contains log terms
                  if "log(" in str(expr):
                      # This is already a log equation - substitute log variables directly
                      subs = {}
                      
                      # For log terms, substitute with log deviations
                      for j, var in enumerate(self.variables):
                          subs[log(Symbol(f"{var}_t"))] = log_vars_cur[j]
                          subs[log(Symbol(f"{var}_tp1"))] = log_vars_fwd[j]
                      
                      # Handle shocks
                      for j, shock in enumerate(shock_names):
                          subs[Symbol(shock)] = shocks[j]
                      
                      subs.update(parameters)
                      
                  else:
                      # Level equation - substitute actual variable values
                      subs = {Symbol(f"{var}_t"): vars_cur[j] for j, var in enumerate(self.variables)}
                      subs.update({Symbol(f"{var}_tp1"): vars_fwd[j] for j, var in enumerate(self.variables)})
                      
                      # Handle shocks
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
                          print(f"Error evaluating equation {i+1}: {eq}, expr: {expr}, error: {e}")
                      residuals[i] = np.nan
              
              if debug:
                  print(f"Log-linear residuals (fwd={log_vars_fwd}, cur={log_vars_cur}, shocks={shocks}): {residuals}")
              return residuals

          log_ss = np.log(e_s)
          print("log_ss",log_ss)
          log_shocks_ss = np.zeros(len(shock_names))
          
          psi_fwd = lambda log_fwd: psi(log_fwd, log_ss, log_shocks_ss)
          psi_cur = lambda log_cur: psi(log_ss, log_cur, log_shocks_ss)
          psi_shocks = lambda log_shocks: psi(log_ss, log_ss, log_shocks)
          
          A_num = self._approx_fprime(log_ss, psi_fwd)
          B_num = self._approx_fprime(log_ss, psi_cur)
          C_num = self._approx_fprime(log_shocks_ss, psi_shocks)
          
      else:
          def psi(vars_fwd, vars_cur, shocks=None):
              residuals = np.zeros(len(self.equations_list))
              shocks = shocks if shocks is not None else np.zeros(len(shock_names))
              
              for i, eq in enumerate(self.equations_list):
                  tp1_terms, t_terms, shock_terms = self._parse_equations(eq)
                  if debug:
                      print(f"Equation {i+1}: {eq}, shock_terms: {shock_terms}")
                  
                  subs = {Symbol(f"{var}_t"): vars_cur[j] for j, var in enumerate(self.variables)}
                  subs.update({Symbol(f"{var}_tp1"): vars_fwd[j] for j, var in enumerate(self.variables)})
                  
                  # Handle shocks - substitute shock symbols directly
                  for j, shock in enumerate(shock_names):
                      subs[Symbol(shock)] = shocks[j]
                  
                  subs.update(parameters)
                  
                  # Correct expression: tp1_terms - t_terms - shock_terms
                  expr = tp1_terms - t_terms - shock_terms
                  
                  if debug:
                      print(f"Equation {i+1} expr before subs: {expr}")
                  
                  expr = expr.subs(subs)
                  
                  try:
                      residuals[i] = float(expr)
                  except (ValueError, TypeError) as e:
                      if debug:
                          print(f"Error evaluating equation {i+1}: {eq}, expr: {expr}, error: {e}")
                      residuals[i] = np.nan
              
              if debug:
                  print(f"Non-log-linear residuals (fwd={vars_fwd}, cur={vars_cur}, shocks={shocks}): {residuals}")
              return residuals

          psi_fwd = lambda fwd: psi(fwd, e_s)
          psi_cur = lambda cur: -psi(e_s, cur)
          psi_shocks = lambda shocks: -psi(e_s, e_s, shocks)
          
          A_num = self._approx_fprime(e_s, psi_fwd)
          B_num = self._approx_fprime(e_s, psi_cur)
          C_num = self._approx_fprime(np.zeros(len(shock_names)), psi_shocks)
          
          # Check residuals for perturbed shocks
          if debug:
              for j, shock in enumerate(shock_names):
                  eps = 1e-6
                  shocks_pert = np.zeros(len(shock_names))
                  shocks_pert[j] = eps
                  residuals_pert = psi(e_s, e_s, shocks_pert)
                  residuals_base = psi(e_s, e_s, np.zeros(len(shock_names)))
                  deriv = (residuals_pert - residuals_base) / eps
                  print(f"Shock {shock} perturbation: residuals_pert={residuals_pert}, residuals_base={residuals_base}")
                  print(f"Shock {shock} numerical derivative: {deriv}")

      self.A_num = A_num
      self.B_num = B_num
      self.C_num = C_num
      
      print("Numerical Jacobian A:\n", A_num)
      print("Numerical Jacobian B:\n", B_num)
      print("Numerical Jacobian C:\n", C_num)
      
      if np.allclose(C_num, 0) and shock_names:
          print("Warning: C_num matrix is all zeros despite shocks. Check equation specifications or _parse_equation logic.")

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
          raise ValueError("Steady state not computed. Call compute_ss() first.")
      if self.approximated==True:
         raise ValueError("The system is already approximated.")
      # Default to analytical method if not specified
      method = method.lower() if method else 'analytical'

      if method == 'analytical':
          A, B, C = self._Analytical_jacobians(debug=debug)
      elif method == 'numerical':
          A, B, C = self._Numerical_jacobians(debug=debug)
      else:
          raise ValueError("Method must be 'analytical' or 'numerical'.")

      # Store Jacobians
      self.A = A
      self.B = B
      self.C = C
      self.approximated = True

      if debug:
          print(f"Approximation ({self.approximation}) completed with method: {method}")
          print(f"Jacobian A:\n{A}")
          print(f"Jacobian B:\n{B}")
          print(f"Jacobian C:\n{C}")

      return A, B, C

  def solve_RE_model(self, Parameters=None,debug=False):
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
            raise ValueError("Model not approximated. Call approximate() first.")

        # Update parameters and compute Jacobians if provided
        if Parameters is not None:
            self.parameters = Parameters
            A, B, C = self._Analytical_jacobians(debug=debug)
        else:
            A, B, C = self.A, self.B, self.C
        # Validate Jacobians
        if A is None or B is None or C is None:
            raise ValueError("Jacobians A, B, C must be provided or computed via approximate().")

        # Validate model dimensions
        if self.n_states == 0 or self.n_controls == 0:
            raise ValueError("Model must have states and controls defined.")

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
                print(f"Matrix shapes: A={A.shape}, B={B.shape}, C={C.shape if C is not None else None}")
                print(f"Variable order: {self.variables[:nk]} (states) + {self.variables[nk:]} (controls)")

            # QZ decomposition
            try:
                S, T, alpha, beta, Q, Z = ordqz(A, B, sort='ouc', output='complex')
            except Exception as e:
                raise ValueError(f"QZ decomposition failed: {e}")

            # Eigenvalue analysis
            eigenvals = np.abs(beta / alpha)
            if debug:
                print(f"QZ eigenvalues (should have {nk} stable): {eigenvals}")

            # Partition QZ results
            z11 = Z[:nk, :nk]
            z21 = Z[nk:, :nk]
            s11 = S[:nk, :nk]
            t11 = T[:nk, :nk]

            # Invertibility check
            if np.linalg.matrix_rank(z11) < nk:
                raise ValueError("Invertibility condition violated: z11 is singular")

            # Stability check
            stable_count = sum(eigenvals < 1)
            if debug:
                print(f"Stable eigenvalues: {stable_count}/{nk} (should be {nk})")
            if stable_count != nk:
                print("Warning: Blanchard-Kahn conditions may not be satisfied")

            # Compute policy and transition functions
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
              warnings.warn('Steady state contains zeros so normalize set to False.', stacklevel=2)
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
              self.irfs[shock] = irfs[[shock_col] + var_cols].rename(columns={shock_col: shock})
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
            raise ValueError("irfs must be a non-empty dictionary of Pandas DataFrames.")

        # Determine shocks to plot
        if shock_names is None:
            shock_names = list(self.irfs.keys())
        else:
            for sh in shock_names:
                if sh not in self.irfs:
                    raise ValueError(f"Shock '{sh}' not found in irfs dictionary.")

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
                raise ValueError(f"No corresponding exogenous state for shock '{sh}'.")
            exo_state = self.exo_states[i]
            exo_col = get_col_name(exo_state)
            
            # List of other variables to plot
            other_cols = [get_col_name(v) for v in self.variables if v != exo_state]

            # Verify columns exist in DataFrame
            if exo_col not in df.columns:
                raise ValueError(f"Column '{exo_col}' not found for shock '{sh}'.")
            missing_cols = [col for col in other_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found for shock '{sh}'.")

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
            ax1.plot(range(T_plot), df_scaled[exo_col], lw=lw, alpha=alpha, label=exo_state)
            ax1.legend(loc='upper right')

            # Subplot 2: IRFs of all other variables
            ax2.set_title(f"{title_prefix}: Other Variables ({sh})")
            ax2.set_xlabel("Time")
            ax2.set_ylabel(ylabel)
            ax2.grid(True)
            for col in other_cols:
                var_name = get_var_from_col(col)
                ax2.plot(range(T_plot), df_scaled[col], lw=lw, alpha=alpha, label=var_name)
            ax2.legend(loc='upper right')

            # Adjust y-axis limits for clarity
            max_y1 = df_scaled[exo_col].max() * 1.1
            min_y1 = df_scaled[exo_col].min() * 1.1
            ax1.set_ylim(min_y1 if min_y1 < 0 else 0, max_y1 if max_y1 > 0 else 1)
            ax1.set_xlim(0, T_plot-1)

            max_y2 = df_scaled[other_cols].max().max() * 1.1
            min_y2 = df_scaled[other_cols].min().min() * 1.1
            ax2.set_ylim(min_y2 if min_y2 < 0 else 0, max_y2 if max_y2 > 0 else 1)
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
            raise ValueError(f"covariance_matrix must be {n_shocks}x{n_shocks}")

    # Generate shocks
    rng = np.random.default_rng(seed)
    eps = rng.multivariate_normal(np.zeros(n_shocks), covariance_matrix, drop_first + T)

    # Initialize state and control arrays
    s = np.zeros((drop_first + T + 1, n_states))  # States from t=0 to t=drop_first + T
    u = np.zeros((drop_first + T, n_costates))    # Controls from t=0 to t=drop_first + T - 1

    # Create shock impact matrix (B): maps shocks to exogenous states
    B = np.zeros((n_states, n_shocks))
    for i, shock in enumerate(self.shocks):
        if i < n_exo_states:
            exo_state_idx = self.states.index(self.exo_states[i])
            B[exo_state_idx, i] = 1.0

    # Simulate dynamics
    for t in range(drop_first + T):
        s[t + 1] = self.p @ s[t] + B @ eps[t]  # Next state
        u[t] = self.f @ s[t + 1]               # Control based on next state

    # Compute deviations (s[t+1], u[t]) for t=drop_first to t=drop_first + T - 1
    sim_deviations = np.hstack((s[drop_first + 1:drop_first + T + 1], u[drop_first:drop_first + T]))

    # Variable columns
    var_cols = [f"{v}_t" for v in self.variables]

    # Steady state values
    ss_values = np.array([self.steady_state[v] for v in self.variables])

    # Check for normalization
    if normalize and not self.approximation=='log_linear' and np.any(np.isclose(ss_values, 0)):
        warnings.warn('Steady state contains zeros so normalize set to False.', stacklevel=2)
        normalize = False

    # Transform based on center and normalize
    if center:
        sim_out = sim_deviations
    else:
        if self.approximation=="log_linear":
            sim_out = sim_deviations + np.log(ss_values)  # Log levels
        else:
            sim_out = sim_deviations + ss_values          # Levels

    if not self.approximation=="log_linear" and normalize:
        sim_out = sim_out / ss_values

    # Create DataFrame with simulated variables
    sim_df = pd.DataFrame(sim_out, columns=var_cols)

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
        raise ValueError("self.simulated must be a pandas DataFrame containing simulation results.")

    # Identify columns
    shock_cols = [f"{sh}_t" for sh in self.shocks]
    exo_state_cols = [f"{s}_t" for s in self.exo_states if f"{s}_t" in self.simulated.columns]
    endo_cols = [f"{v}_t" for v in self.variables if v not in self.exo_states and f"{v}_t" in self.simulated.columns]

    if not exo_state_cols and not endo_cols:
        raise ValueError("No exogenous states or endogenous variables found in self.simulated.")

    # Number of subplots: one for each exogenous state + one for endogenous variables
    n_exo = len(exo_state_cols)
    n_subplots = n_exo + 1 if endo_cols else n_exo

    # Create figure and subplots
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    # Plot exogenous states
    for i, col in enumerate(exo_state_cols):
        axes[i].plot(self.simulated.index, self.simulated[col], label=col, color=f"C{i}")
        axes[i].set_ylabel(col)
        axes[i].legend(loc="upper right")
        axes[i].grid(True)

    # Plot endogenous variables together
    if endo_cols:
        for col in endo_cols:
            axes[n_exo].plot(self.simulated.index, self.simulated[col], label=col)
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
from abc import ABC, abstractmethod

class nonlinear_dsge(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def residual(self, coeffs, state, basis, policy_func):
        pass

    @abstractmethod
    def initial_guess(self, basis):
        pass


