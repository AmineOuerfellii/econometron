from sympy import symbols, Symbol, Matrix , collect , S, exp ,log
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application 
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import ordqz,qz
import matplotlib.animation as animation
import warnings

# Time symbol
t = Symbol('t', integer=True)

# Transformations for parsing equations
_transformations = standard_transformations + (implicit_multiplication_application,)

class Model:
    """A class to solve and analyze a simplified RBC model."""
    def __init__(
        self,
        equations=None,
        variables=None,
        states=None,
        exo_states=None,
        endo_states=None,
        shock_names=None,
        parameters=None,
        shock_prefix=None,
        n_states=None,
        n_exo_states=None,
        log_linear=False,
        shock_variance=None,
        steady_state=None
    ):
        self.equations_list = equations  # Renamed to avoid confusion with equations method
        self.names = {'variables': variables}  # For compute_ss compatibility
        self.variables = variables
        self.states = states
        self.exo_states = exo_states
        self.endo_states = endo_states
        self.shock_names = shock_names
        self.parameters = parameters
        self.shock_prefix = shock_prefix or ''
        self.n_states = n_states or (len(states) if states else None)
        self.n_exo_states = n_exo_states or (len(exo_states) if exo_states else None)
        self.n_vars = len(variables)
        self.ss = steady_state
        self.linearized_system = None
        self.log_linear=log_linear
        self.f = None
        self.p = None
        self.simulated = None
        self.irfs = None
        self.shock_variance = shock_variance or {shock: 0.01**2 for shock in shock_names}


    def set_initial_guess(self, initial_guess):
        if len(initial_guess) != len(self.variables):
            raise ValueError("Initial guess must match the number of variables.")
        self.initial_guess = np.array(initial_guess)


    def _parse_equation(self, eq):
        local_dict = {}
        for var in self.variables:
            local_dict[f"{var}_t"] = Symbol(f"{var}_t")
            local_dict[f"{var}_tp1"] = Symbol(f"{var}_tp1")
            local_dict[f"{var}_tm1"] = Symbol(f"{var}_tm1")
        for shock in self.shock_names:
            local_dict[f"{shock}_t"] = Symbol(f"{shock}_t")
        for param in self.parameters:
            local_dict[param] = Symbol(param)

        eq_normalized = eq.replace("{t+1}", "tp1").replace("{t-1}", "tm1").replace("_t", "_t")
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
            has_shock = any(f"{shock}_t" in term_str for shock in self.shock_names)
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
                                    [local_dict[f"{shock}_t"] for shock in self.shock_names], evaluate=False)
                for sym, coeff in coeff_dict.items():
                    sym_str = str(sym)
                    if 'tp1' in sym_str:
                        tp1_terms += coeff * sym
                    elif any(f"{shock}_t" in sym_str for shock in self.shock_names):
                        shock_terms += coeff * sym
                    else:
                        t_terms += coeff * sym
                if Symbol('1') in coeff_dict:
                    shock_terms += coeff_dict[Symbol('1')]

        return -tp1_terms, -t_terms, shock_terms

    def equations(self, vars_t_plus_1, vars_t, parameters):
        """Evaluate residuals of the model equations for compute_ss."""
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
        for shock in self.shock_names:
            subs[Symbol(f"{shock}_t")] = parameters.get(shock, 0.0)
        subs.update({Symbol(k): float(v) for k, v in parameters.items()})

        for i, eq in enumerate(self.equations_list):
            tp1_terms, t_terms, shock_terms = self._parse_equation(eq)
            # print(f"Parsed equation {i+1}: t+1={tp1_terms}, t={t_terms}, shocks={shock_terms}")
            
            # Residual: LHS_{t+1} - RHS_t - shocks
            residual = (tp1_terms + t_terms - shock_terms).subs(subs)
            try:
                residual_value = float(residual.evalf().as_real_imag()[0])
                residuals.append(residual_value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannot convert residual to float for equation '{eq}': {residual}")

        return np.array(residuals)

    def compute_ss(self, guess=None, method='fsolve', options={}):
        if guess is None:
            guess = np.zeros(self.n_vars)  # Changed to zeros for NK model
        else:
            if isinstance(guess, pd.Series):
                guess = guess.loc[self.names['variables']].values
            elif isinstance(guess, list):
                guess = np.array(guess)
            guess = np.array(guess, dtype=float)

        def ss_fun(variables):
            # In steady state, vars_t_plus_1 = vars_t
            return self.equations(variables, variables, self.parameters)

        if not np.iscomplexobj(self.parameters) and not np.iscomplexobj(guess):
            if method == 'fsolve':
                steady_state = fsolve(ss_fun, guess, **options)
            else:
                raise ValueError("Only 'fsolve' is implemented for this example.")
        else:
            raise ValueError("Complex parameters not supported in this model.")

        self.ss = pd.Series(steady_state, index=self.names['variables'])
        residuals = ss_fun(steady_state)
        print("Steady-state residuals:", residuals)
        if np.any(np.abs(residuals) > 1e-8):
            print("Warning: Large steady-state residuals detected. Check equations or initial guess.")
        
        return self.ss


    def analytical_jacobians(self):
        vars_t = [Symbol(f"{var}_t") for var in self.variables]
        vars_tp1 = [Symbol(f"{var}_tp1") for var in self.variables]

        A = np.zeros((len(self.equations_list), len(self.variables)))
        B = np.zeros((len(self.equations_list), len(self.variables)))

        subs = {Symbol(f"{var}_t"): self.ss[var] for var in self.variables}
        subs.update({Symbol(f"{var}_tp1"): self.ss[var] for var in self.variables})
        subs.update({Symbol(k): v for k, v in self.parameters.items()})
        # print("Substitution dictionary:", subs)

        if self.log_linear:
            e_s = np.array([self.ss[var] for var in self.variables])
            if np.any(np.isclose(e_s, 0)):
                raise ValueError("Steady state contains zeros; cannot compute log-linear Jacobians.")
            log_vars_t = [Symbol(f"log_{var}_t") for var in self.variables]
            log_vars_tp1 = [Symbol(f"log_{var}_tp1") for var in self.variables]
            eqs = []
            for eq in self.equations_list:
                tp1_terms, t_terms, _ = self._parse_equation(eq)
                expr = tp1_terms + t_terms
                subs_log = {}
                for var, log_var, log_var_tp1 in zip(self.variables, log_vars_t, log_vars_tp1):
                    subs_log[Symbol(f"{var}_t")] = exp(log_var)
                    subs_log[Symbol(f"{var}_tp1")] = exp(log_var_tp1)
                expr = expr.subs(subs_log)
                expr = expr + 1
                expr = log(expr)
                eqs.append(expr)
            A_mat = Matrix(eqs).jacobian(log_vars_tp1)
            B_mat = Matrix(eqs).jacobian(log_vars_t)
            log_subs = {Symbol(f"log_{var}_t"): log(self.ss[var]) for var in self.variables}
            log_subs.update({Symbol(f"log_{var}_tp1"): log(self.ss[var]) for var in self.variables})
            log_subs.update({Symbol(k): v for k, v in self.parameters.items()})
            A = np.array(A_mat.subs(log_subs), dtype=float)
            B = - np.array(B_mat.subs(log_subs), dtype=float)
        else:
            for i, eq in enumerate(self.equations_list):
                tp1_terms, t_terms, _ = self._parse_equation(eq)
                # print(f"Equation {i+1}: t+1 terms={tp1_terms}, t terms={t_terms}")
                for j, var in enumerate(vars_tp1):
                    coeff = tp1_terms.diff(var) if tp1_terms != S.Zero else S.Zero
                    A[i, j] = float(coeff.subs(subs)) if coeff != S.Zero else 0.0
                for j, var in enumerate(vars_t):
                    coeff = t_terms.diff(var) if t_terms != S.Zero else S.Zero
                    B[i, j] = - float(coeff.subs(subs)) if coeff != S.Zero else 0.0

        expected_shape = (len(self.equations_list), len(self.variables))
        if A.shape != expected_shape:
            A = A.T
        if B.shape != expected_shape:
            B = B.T
        
        # print("Analytical Jacobian A:\n", A)
        # print("Analytical Jacobian B:\n", B)
        return A, B

    def approx_fprime(self, x, f, epsilon=None):
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


    def approximation_Mat_A_Mat_B(self):
        e_s = np.array([self.ss[var] for var in self.variables])
        A_num = np.zeros((len(self.equations_list), len(self.variables)))
        B_num = np.zeros((len(self.equations_list), len(self.variables)))

        parameters = self.parameters.copy()

        if self.log_linear:
            equilibrium_right = np.ones(len(self.variables))
            def psi(log_vars_fwd, log_vars_cur):
                vars_fwd = np.exp(log_vars_fwd)
                vars_cur = np.exp(log_vars_cur)
                residuals = np.zeros(len(self.equations_list))
                for i, eq in enumerate(self.equations_list):
                    tp1_terms, t_terms, shock_terms = self._parse_equation(eq)
                    subs = {f"{var}_t": vars_cur[j] for j, var in enumerate(self.variables)}
                    subs.update({f"{var}_tp1": vars_fwd[j] for j, var in enumerate(self.variables)})
                    subs.update({f"{shock}_t": parameters.get(shock, 0.0) for shock in self.shock_names})
                    subs.update(parameters)
                    expr = (tp1_terms + t_terms - shock_terms).subs(subs)
                    residuals[i] = np.log(float(expr) + 1) - np.log(equilibrium_right[i])
                return residuals

            log_ss = np.log(e_s + 1e-10)  # Avoid log(0)
            psi_fwd = lambda log_fwd: psi(log_fwd, log_ss)
            psi_cur = lambda log_cur: psi(log_ss, log_cur)
            A_num = self.approx_fprime(log_ss, psi_fwd)  # No negation
            B_num = self.approx_fprime(log_ss, psi_cur)  # No negation
            self.linearized_system = True
        else:
            def psi(vars_fwd, vars_cur):
                residuals = np.zeros(len(self.equations_list))
                for i, eq in enumerate(self.equations_list):
                    tp1_terms, t_terms, shock_terms = self._parse_equation(eq)
                    subs = {f"{var}_t": vars_cur[j] for j, var in enumerate(self.variables)}
                    subs.update({f"{var}_tp1": vars_fwd[j] for j, var in enumerate(self.variables)})
                    subs.update({f"{shock}_t": parameters.get(shock, 0.0) for shock in self.shock_names})
                    subs.update(parameters)
                    expr = (tp1_terms + t_terms - shock_terms).subs(subs)
                    residuals[i] = float(expr)
                return residuals

            psi_fwd = lambda fwd: psi(fwd, e_s)
            psi_cur = lambda cur: psi(e_s, cur)
            A_num = self.approx_fprime(e_s, psi_fwd)  # No negation
            B_num = self.approx_fprime(e_s, psi_cur) 
            self.linearized_system = False

        self.A_num = A_num
        self.B_num = B_num
        print("Numerical Jacobian A:\n", A_num)
        print("Numerical Jacobian B:\n", B_num)
        return A_num, B_num


    def approximate(self):
        A, B = self.approximation_Mat_A_Mat_B()
        self.linearized_system = []
        for i in range(len(self.equations_list)):
            terms = {}
            for j, var in enumerate(self.variables):
                if abs(A[i, j]) > 1e-10:
                    coeff = A[i, j] * self.ss[var] if not self.log_linear else A[i, j]
                    terms[Symbol(f"hat_{var}_tp1")] = coeff
                if abs(B[i, j]) > 1e-10:
                    coeff = B[i, j] * self.ss[var] if not self.log_linear else B[i, j]
                    terms[Symbol(f"{var}_t")] = coeff
            if i == len(self.equations_list) - 1:
                for shock in self.shock_names:
                    terms[Symbol(f"{shock}_t")] = -1.0  # Correct shock coefficient
            self.linearized_system.append(terms)
            print(f"Linearized equation {i+1}:", terms)


    def solve(self,parameters):
      self.parameter=parameters

      A,B=self.analytical_jacobians()
      return self.solve_model(A,B, len(self.states))



    def solve_model(self,a, b, nk):
        """
        Solve the linear rational expectations model using QZ decomposition.

        Parameters:
        a (ndarray): Coefficient matrix A
        b (ndarray): Coefficient matrix B
        nk (int): Number of predetermined (stable) variables

        Returns:
        f (ndarray): Forward-looking component
        p (ndarray): Law of motion matrix for the predetermined variables
        """


        # Step 2: Sort eigenvalues so that stable ones (|eig| < 1) come first
        sort_fun = lambda alpha, beta: (np.abs(beta / alpha) < 1) & (alpha != 0)

        s, t, alpha, beta, q, z = ordqz(a, b, sort=sort_fun, output='complex')

        # Step 3: Extract relevant submatrices
        z21 = z[nk:, :nk]
        z11 = z[:nk, :nk]

        # Step 4: Check invertibility
        if np.linalg.matrix_rank(z11) < nk:
            raise ValueError("Invertibility condition violated")

        z11i = np.linalg.inv(z11)
        s11 = s[:nk, :nk]
        t11 = t[:nk, :nk]

        # Step 5: Optional check on eigenvalues
        if (np.abs(t[nk-1, nk-1]) > np.abs(s[nk-1, nk-1]) or
            np.abs(t[nk, nk]) < np.abs(s[nk, nk])):
            print("Warning: Wrong number of stable eigenvalues.")

        # Step 6: Compute dynamics
        dyn = np.linalg.solve(s11, t11)

        f = np.real(z21 @ z11i)
        p = np.real(z11 @ dyn @ z11i)
        self.f=f
        self.p=p
        return f, p


    def simulate(self, T=51, drop_first=300, covariance_matrix=None, seed=None):
        """
        Simulate the DSGE model dynamics.

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

        Returns:
        --------
        None
            Sets self.simulated to a DataFrame with shocks and simulated variables.
        """
        if self.f is None or self.p is None:
            raise ValueError("Model must be solved before simulation.")

        n_states = self.n_states
        n_costates = len(self.variables) - n_states
        n_shocks = len(self.shock_names)
        n_exo_states = self.n_exo_states

        # Set covariance matrix
        if covariance_matrix is None:
            # Use shock_variance attribute rather than parameters
            variances = [self.shock_variance.get(shock, 0.01**2) for shock in self.shock_names]
            covariance_matrix = np.diag(variances)
        else:
            covariance_matrix = np.array(covariance_matrix)
            if covariance_matrix.shape != (n_shocks, n_shocks):
                raise ValueError(f"covariance_matrix must be {n_shocks}x{n_shocks}")

        # Generate shocks
        rng = np.random.default_rng(seed)
        eps = rng.multivariate_normal(np.zeros(n_shocks), covariance_matrix, drop_first + T)

        # Initialize state and control arrays
        s = np.zeros((drop_first + T + 1, n_states))  # States
        u = np.zeros((drop_first + T, n_costates))    # Controls

        # Create shock impact matrix (B): maps shocks to exogenous states
        B = np.zeros((n_states, n_shocks))
        
        # For exogenous states, each shock affects the corresponding state
        for i, shock in enumerate(self.shock_names):
            if i < n_exo_states:  # Ensure we don't exceed exo states
                # Find the corresponding exogenous state
                exo_state_idx = self.states.index(self.exo_states[i])
                B[exo_state_idx, i] = 1.0  # Direct impact of shock on exogenous state

        # Simulate dynamics
        for t in range(drop_first + T):
            s[t+1] = self.p @ s[t] + B @ eps[t]  # State transition
            u[t] = self.f @ s[t]                 # Control variables

        # Prepare output
        sim = np.hstack((s[drop_first:drop_first + T], u[drop_first:]))
        cols = [f"{'hat_' if self.log_linear else ''}{v}_t" for v in self.variables]

        # Transform to levels if needed
        ss_values = np.array([self.ss[v] for v in self.variables])
        
        if self.log_linear:
            # For log-linear model, convert log-deviations to levels
            sim_levels = np.exp(sim) * ss_values
            sim_out = sim_levels
        else:
            # For linear model, add steady state to deviations
            sim_out = sim + ss_values

        self.simulated = pd.concat([
            pd.DataFrame(eps[drop_first:], columns=[f"{sh}_t" for sh in self.shock_names]),
            pd.DataFrame(sim_out, columns=cols)
        ], axis=1)
        
        return self.simulated


    def compute_irfs(self, T=41, t0=1, shocks=None, center=True, normalize=True):
        """
        Simulate impulse response functions (IRFs).

        Args:
            T (int): Number of periods to simulate. Default: 41
            t0 (int): Period for shock realization (0-based indexing). Default: 1
            shocks (dict, pd.Series, list, ndarray): Shock names and values. Default: 0.01 for all shocks
            center (bool): If True, returns deviations from steady state. Default: True
            normalize (bool): If True and not log-linear, divide by steady states. Default: True

        Sets:
            self.irfs: Dictionary of Pandas DataFrames with IRF data
        """
        if self.f is None or self.p is None:
            raise ValueError("Model must be solved before computing IRFs.")

        if not self.shock_names:
            raise ValueError("self.shock_names is empty.")

        if normalize and np.any(np.isclose(self.ss, 0)):
            normalize = False
            warnings.warn("Steady state contains zeros; normalize set to False.", stacklevel=2)

        # Process shocks
        if shocks is None:
            shocks = pd.Series(0.01, index=self.shock_names)
        elif isinstance(shocks, dict):
            shocks = pd.Series(shocks)
        elif isinstance(shocks, (list, np.ndarray)):
            shocks = pd.Series(shocks, index=self.shock_names[:len(shocks)])
        elif not isinstance(shocks, pd.Series):
            raise TypeError("Shocks must be a Series, dict, list, or ndarray.")

        self.irfs = {}
        n_exo_states = self.n_exo_states

        # Create shock impact matrix (B): maps shocks to exogenous states
        B = np.zeros((self.n_states, len(self.shock_names)))
        
        # For exogenous states, each shock affects the corresponding state
        for i, shock in enumerate(self.shock_names):
            if i < n_exo_states:  # Ensure we don't exceed exo states
                # Find the corresponding exogenous state
                exo_state_idx = self.states.index(self.exo_states[i])
                B[exo_state_idx, i] = 1.0  # Direct impact of shock on exogenous state

        for sh, val in shocks.items():
            if sh not in self.shock_names:
                warnings.warn(f"{sh} not in self.shock_names.", stacklevel=2)
                continue

            # Initialize eps
            eps = np.zeros((T, len(self.shock_names)))
            shock_idx = self.shock_names.index(sh)
            if t0 < T:
                eps[t0, shock_idx] = val

            # Simulate
            s = np.zeros((T + 1, self.n_states))
            u = np.zeros((T, len(self.variables) - self.n_states))

            # Apply the shock to exogenous states
            for i in range(T):
                if i == t0:
                    # Apply shock impact at t0
                    print(B)
                    s[i+1] = self.p @ s[i] + B @ eps[i]
                else:
                    # Normal dynamics without shock
                    s[i+1] = self.p @ s[i]
                u[i] = self.f @ s[i]

            s = s[1:]  # Drop initial state
              
            # Combine states and controls
            sim = np.hstack((s, u))
            
            # Create var names for columns
            var_cols = [f"{'hat_' if self.log_linear else ''}{v}_t" for v in self.variables]
            sim_df = pd.DataFrame(sim, columns=var_cols)

            # Transform to levels if needed
            ss_values = np.array([self.ss[v] for v in self.variables])
            
            if self.log_linear:
                if center:
                    # Keep as log-deviations
                    pass
                else:
                    # Convert to levels
                    sim_df = pd.DataFrame(np.exp(sim) * ss_values, columns=var_cols)
            else:
                if center:
                    # Keep as deviations
                    pass
                else:
                    # Add steady state
                    sim_df = pd.DataFrame(sim + ss_values, columns=var_cols)
                
                if normalize and not center:
                    # Normalize by steady state (only for non-log, non-centered IRFs)
                    sim_df = pd.DataFrame(sim_df.values / ss_values, columns=var_cols)

            # Build output
            data = np.concatenate((eps, sim_df.values), axis=1)
            cols = [f"{name}_t" for name in self.shock_names] + sim_df.columns.tolist()
            self.irfs[sh] = pd.DataFrame(data, columns=cols)

        return self.irfs



    def _plot_simulation(self):
        if self.simulated is None:
            print("No simulation data to plot.")
            return
        plt.figure(figsize=(14, 8))
        for col in self.simulated.columns:
            plt.plot(self.simulated[col], label=col)
        plt.title("Simulation Results")
        plt.xlabel("Time")
        plt.ylabel("Deviation from Steady State" if self.log_linear else "Level")
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()







    def plot_irfs(self, vars_first_subplot=None, var_second_subplot=None, shock_names=None, 
                  animate=False, scale=100, figsize=(12, 4), lw=5, alpha=0.5, 
                  title_prefix="IRF", ylabel="Percentage Deviation", frames_per_period=20):
        """
        Plot impulse response functions (IRFs) in two subplots with optional animation.

        Args:
            vars_first_subplot (list, optional): Variables for first subplot. Defaults to all variables.
            var_second_subplot (str, optional): Variable for second subplot (with shock). Defaults to first variable.
            shock_names (list, optional): Shocks to plot. Defaults to all shocks in irfs.
            animate (bool, optional): If True, animates the plots. Default: False.
            scale (float, optional): Scaling factor for IRF values. Default: 100.
            figsize (tuple, optional): Figure size. Default: (12, 4).
            lw (float, optional): Line width. Default: 5.
            alpha (float, optional): Line transparency. Default: 0.5.
            title_prefix (str, optional): Title prefix. Default: "IRF".
            ylabel (str, optional): Y-axis label. Default: "Percentage Deviation".
            frames_per_period (int, optional): Frames per period for animation. Default: 20.

        Returns:
            None or animation object (if animate=True).
        """
        if not isinstance(self.irfs, dict) or not self.irfs:
            raise ValueError("irfs must be a non-empty dictionary of Pandas DataFrames.")

        if shock_names is None:
            shock_names = list(self.irfs.keys())
        else:
            for sh in shock_names:
                if sh not in self.irfs:
                    raise ValueError(f"Shock '{sh}' not found in irfs dictionary.")

        for sh in shock_names:
            df = self.irfs[sh]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"irfs['{sh}'] must be a Pandas DataFrame.")

            all_columns = df.columns.tolist()
            shock_col = f"{sh}_t"
            if shock_col not in all_columns:
                raise ValueError(f"Shock column '{shock_col}' not found for shock '{sh}'.")

            var_columns = [col for col in all_columns if col not in [f"{s}_t" for s in self.shock_names]]
            if not var_columns:
                raise ValueError(f"No variable columns found for shock '{sh}'.")

            if vars_first_subplot is None:
                vars_to_plot = var_columns
            else:
                vars_to_plot = [v for v in vars_first_subplot if v in var_columns]
                if not vars_to_plot:
                    raise ValueError(f"No valid variables in vars_first_subplot for shock '{sh}'.")

            if var_second_subplot is None:
                var_to_plot = var_columns[0]
            else:
                var_to_plot = var_second_subplot if var_second_subplot in var_columns else var_columns[0]

            # Scale the data
            df_scaled = df * scale
            T = len(df)

            # Set up figure and subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Subplot 1: Selected variables
            ax1.set_title(f"{title_prefix}: Variables ({sh})")
            ax1.set_xlabel("Time")
            ax1.set_ylabel(ylabel)
            ax1.grid(True)

            # Subplot 2: Shock and selected variable
            ax2.set_title(f"{title_prefix}: Shock and Variable ({sh})")
            ax2.set_xlabel("Time")
            ax2.set_ylabel(ylabel)
            ax2.grid(True)

            # Plot setup
            lines1 = [ax1.plot([], [], lw=lw, alpha=alpha, label=var)[0] for var in vars_to_plot]
            lines2 = [ax2.plot([], [], lw=lw, alpha=alpha, label=var)[0] for var in [shock_col, var_to_plot]]
            time_line1 = ax1.axvline(0, color='k', linestyle='--', alpha=0.3)
            time_line2 = ax2.axvline(0, color='k', linestyle='--', alpha=0.3)

            # Set axis limits
            max_y1 = df_scaled[vars_to_plot].max().max() * 1.1
            min_y1 = df_scaled[vars_to_plot].min().min() * 1.1
            ax1.set_ylim(min_y1 if min_y1 < 0 else 0, max_y1 if max_y1 > 0 else 1)
            ax1.set_xlim(0, T-1)

            max_y2 = df_scaled[[shock_col, var_to_plot]].max().max() * 1.1
            min_y2 = df_scaled[[shock_col, var_to_plot]].min().min() * 1.1
            ax2.set_ylim(min_y2 if min_y2 < 0 else 0, max_y2 if max_y2 > 0 else 1)
            ax2.set_xlim(0, T-1)

            ax1.legend(loc='upper right', ncol=len(vars_to_plot))
            ax2.legend(loc='upper right', ncol=2)

            if animate:
                def update(frame):
                    t = frame // frames_per_period
                    subframe = frame % frames_per_period
                    if subframe == 0 and t < T:
                        for i, var in enumerate(vars_to_plot):
                            x = range(t + 1)
                            y = df_scaled[var].iloc[:t + 1]
                            lines1[i].set_data(x, y)
                        for i, var in enumerate([shock_col, var_to_plot]):
                            x = range(t + 1)
                            y = df_scaled[var].iloc[:t + 1]
                            lines2[i].set_data(x, y)
                        time_line1.set_xdata([t, t])
                        time_line2.set_xdata([t, t])
                    return lines1 + lines2 + [time_line1, time_line2]

                ani = animation.FuncAnimation(
                    fig,
                    update,
                    frames=T * frames_per_period,
                    interval=50,
                    blit=True
                )
            else:
                for i, var in enumerate(vars_to_plot):
                    lines1[i].set_data(range(T), df_scaled[var])
                for i, var in enumerate([shock_col, var_to_plot]):
                    lines2[i].set_data(range(T), df_scaled[var])
                time_line1.set_visible(False)
                time_line2.set_visible(False)

            plt.tight_layout()
            plt.show()
            if animate:
                return ani