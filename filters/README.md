# Filters Module — API & Usage Guide

This document provides a comprehensive overview of the filters in the `filters/` directory of the Econometron library. It is intended for economists, data scientists, and students working with state-space models, Kalman filtering, and likelihood-based estimation.

---

## File: `Kalman.py`

### Class: `Kalman`
Implements the Kalman filter and smoother for state-space models.

#### Constructor: `__init__(params)`
- **Purpose:** Initialize the Kalman filter with model parameters.
- **Arguments:**
  - `params`: Dictionary with keys:
    - `A`: State transition matrix (n x n)
    - `D`: Observation matrix (m x n)
    - `Q`: State covariance matrix (n x n)
    - `R`: Observation covariance matrix (m x m)
    - `x0`: Initial state estimate (n x 1, optional)
    - `P0`: Initial state covariance (n x n, optional)
- **Role:** Sets up the filter for subsequent filtering or smoothing.

#### Method: `filter(y)`
- **Purpose:** Run the Kalman filter on observed data.
- **Arguments:**
  - `y`: Observations (m x T)
- **Returns:** Dictionary with filtered states, covariances, residuals, and log-likelihood.
- **Example:**
```python
params = {...}  # see above
kf = Kalman(params)
result = kf.filter(y)
print(result['log_lik'])
```

#### Method: `smooth(y)`
- **Purpose:** Run the Kalman smoother on observed data.
- **Arguments:**
  - `y`: Observations (m x T)
- **Returns:** Dictionary with smoothed states, predicted states, covariances, and residuals.
- **Example:**
```python
result = kf.smooth(y)
print(result['Xsm'])
```

---

## File: `objective_kal.py`

### Function: `kalman_objective(params, fixed_params, param_names, y, update_state_space)`
- **Purpose:** Objective function for parameter estimation using the Kalman filter (e.g., for MLE or genetic algorithms).
- **Arguments:**
  - `params`: Parameters to optimize (array-like)
  - `fixed_params`: Dictionary of fixed parameters
  - `param_names`: List of parameter names to optimize
  - `y`: Observed data (m x T)
  - `update_state_space`: Function to update state-space matrices given parameters
- **Returns:** Negative log-likelihood (float)
- **Calls:**
  - `update_state_space` (user-supplied)
  - `Kalman.filter`
- **Example:**
```python
neg_loglik = kalman_objective(params, fixed_params, param_names, y, update_state_space)
```

---

## File: `Kalman_smooth.py`

### Function: `kal_smooth(params, fixed_params, param_names, y, update_state_space)`
- **Purpose:** Objective function for Kalman smoothing, typically used in state estimation or as part of an optimization routine.
- **Arguments:**
  - `params`: Parameters to optimize (array-like)
  - `fixed_params`: Dictionary of fixed parameters
  - `param_names`: List of parameter names to optimize
  - `y`: Observed data (m x T)
  - `update_state_space`: Function to update state-space matrices given parameters
- **Returns:** Smoothed state (array)
- **Calls:**
  - `update_state_space` (user-supplied)
  - `Kalman.smooth`
- **Example:**
```python
smoothed_state = kal_smooth(params, fixed_params, param_names, y, update_state_space)
```

---

## File: `__init__.py`

- Imports and exposes the main filter and objective functions:
  - `Kalman` (class)
  - `kalman_objective` (function)
  - `kal_smooth` (function)
- **Usage:**
```python
from filters import Kalman, kalman_objective, kal_smooth
```

---

## Typical Workflow Example

```python
from filters import Kalman, kalman_objective, kal_smooth

# 1. Define or estimate state-space parameters (A, D, Q, R, etc.)
params = {...}

# 2. Run Kalman filter
kf = Kalman(params)
result = kf.filter(y)
print('Log-likelihood:', result['log_lik'])

# 3. Run Kalman smoother
smoothed = kf.smooth(y)
print('Smoothed states:', smoothed['Xsm'])

# 4. Use kalman_objective in optimization (e.g., for MLE)
neg_loglik = kalman_objective(params_to_optimize, fixed_params, param_names, y, update_state_space)

# 5. Use kal_smooth for state estimation in optimization
smoothed_state = kal_smooth(params_to_optimize, fixed_params, param_names, y, update_state_space)
```

---

## Notes
- All functions are designed to be modular and can be integrated into custom estimation or filtering pipelines.
- The Kalman filter assumes a linear-Gaussian state-space model.
- For more advanced use (e.g., time-varying parameters, non-Gaussian noise), consider extending the base classes or functions.
