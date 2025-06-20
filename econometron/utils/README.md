# Utils Module — API & Usage Guide

This document provides a comprehensive overview of the `utils/` directory in the Econometron library. It is designed for economists, data scientists, and students working with model estimation, optimization, state-space construction, and time series processing.

---

## state_space/update_ss.py

### Function: `make_state_space_updater`
- **Purpose:** Creates a function to update state-space matrices given model parameters.
- **Arguments:**
  - `base_params`: Default model parameters (dict)
  - `solver`: Function to solve the model (e.g., `Model.solve`)
  - `build_R`: Function to build the R matrix
  - `build_C`: Function to build the C matrix
  - `derived_fn`: (Optional) Function to compute derived parameters
- **Returns:** Callable that takes parameter updates and returns state-space matrices (`A`, `D`, `Q`, `R`)
- **Example:**
```python
update_state_space = make_state_space_updater(base_params, solver, build_R, build_C, derived_fn)
ss = update_state_space({'beta': 0.98})
print(ss['A'])
```

---

## optimizers/optim.py

### Function: `simulated_annealing`
- **Purpose:** Simulated annealing optimization for parameter estimation.
- **Arguments:** Objective function, initial guess, bounds, temperature, cooling rate, etc.
- **Returns:** Dictionary with optimal parameters, function value, and diagnostics.
- **Example:**
```python
result = simulated_annealing(func, x0, lb, ub, T, cooling_rate, n_temp, n_steps, seed, max_evals)
print(result['x'], result['fun'])
```

---

## optimizers/eval.py

### Function: `evaluate_func`
- **Purpose:** Evaluates an objective function at given parameters.
- **Arguments:**
  - `function`: Callable
  - `params`: Parameters
- **Returns:** Function value
- **Example:**
```python
val = evaluate_func(lambda x: sum(x), [1,2,3])
```

---

## estimation/OLS.py

### Function: `ols_estimator`
- **Purpose:** Ordinary Least Squares (OLS) estimation with standard errors and p-values.
- **Arguments:**
  - `X`: Regressor matrix
  - `Y`: Response matrix
- **Returns:** Tuple of (beta, residuals, se, z_values, p_values)
- **Example:**
```python
beta, res, se, z, p = ols_estimator(X, Y)
print(beta)
```

---

## estimation/MLE.py

### Function: `genetic_algorithm_kalman`
- **Purpose:** Parameter estimation for DSGE models using a genetic algorithm and Kalman filter.
- **Arguments:** Data, initial guess, bounds, parameter names, fixed params, state-space updater, and GA settings.
- **Returns:** Dictionary with optimized parameters and results table.
- **Example:**
```python
result = genetic_algorithm_kalman(y, x0, lb, ub, param_names, fixed_params, update_state_space)
print(result['result'])
```

### Function: `simulated_annealing_kalman`
- **Purpose:** Parameter estimation using simulated annealing and Kalman filter.
- **Arguments:** Similar to above, with annealing settings.
- **Returns:** Dictionary with results and table.

---

## estimation/Bayesian.py

### Function: `rwm_kalman`
- **Purpose:** Bayesian estimation (Random Walk Metropolis) for DSGE models using the Kalman filter.
- **Arguments:** Data, initial guess, bounds, parameter names, fixed params, state-space updater, MCMC settings, and prior.
- **Returns:** Dictionary with samples, log-posterior, acceptance rate, and results table.
- **Example:**
```python
result = rwm_kalman(y, x0, lb, ub, param_names, fixed_params, update_state_space)
print(result['result']['samples'])
```

---

## estimation/prior.py

### Function: `make_prior_function`
- **Purpose:** Creates a log-prior function for Bayesian estimation.
- **Arguments:**
  - `param_names`: List of parameter names
  - `priors`: Dict mapping parameter names to (distribution, kwargs)
  - `bounds`: Dict of parameter bounds
  - `verbose`: Print debug info if True
- **Returns:** Callable log-prior function
- **Example:**
```python
prior = make_prior_function(['beta'], {'beta': (scipy.stats.beta, {'a':2, 'b':2})}, {'beta': (0,1)})
print(prior([0.5]))
```

---

## estimation/results.py

### Function: `compute_stats`
- **Purpose:** Computes standard errors and p-values using the numerical Hessian.
- **Arguments:** Parameter estimates, log-likelihood, objective function, etc.
- **Returns:** Dictionary with standard errors and p-values.
- **Example:**
```python
stats = compute_stats(params, log_lik, func)
print(stats['std_err'])
```

### Function: `create_results_table`
- **Purpose:** Creates a results table for optimization or sampling methods.
- **Arguments:** Result dict, parameter names, log-likelihood, objective function, method, etc.
- **Returns:** Table (e.g., pandas DataFrame)

---

## data_preparation/process_timeseries.py

### Function: `process_time_series`
- **Purpose:** Processes time series data: handles missing values, checks stationarity, and visualizes ACF/PACF.
- **Arguments:** DataFrame/Series, date column, columns, max differencing, significance level, plot flag.
- **Returns:** Dictionary with stationarity status, differenced data, and ADF results.
- **Example:**
```python
results = process_time_series(df, date_column='date', columns=['GDP'], plot=True)
print(results['GDP']['stationary'])
```

---

## Notes
- All functions are modular and can be integrated into custom estimation, filtering, or preprocessing pipelines.
- For advanced usage, see the source code for additional arguments and options.
