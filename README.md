# Econometron

**Econometron** is a Python library for building, simulating, and estimating dynamic and Timeseries models, with a focus on:
- **Dynamic Stochastic General Equilibrium (DSGE) models**
- **Vector Autoregression (VAR) models**
- **Vector Autoregression Integrated Moving Average (VARIMA) models**
- **State space models and Kalman filtering**
- **Deep learning time series (N-BEATS)**

Econometron offers a modular, extensible architecture and a suite of tools for economists, researchers, and data scientists working in quantitative macroeconomics.

---

## Features

### DSGE Modelling
- **Flexible Model Definition:** Specify model equations, parameters, and variables in a symbolic, user-friendly syntax.
- **Simulation:** Generate time series data under stochastic shocks (e.g., technology, policy shocks).
- **Estimation:** Calibrate and estimate model parameters using numerical solvers and optimization techniques.
- **Linear & Nonlinear Solvers:** Solve nonlinear or linearized DSGE models efficiently.

### VAR & VARIMA Modelling
- **VAR:** Analyze and forecast multivariate time series (e.g., GDP, inflation, interest rates) with automatic lag selection, diagnostics, and impulse response analysis.
- **VARIMA:** Model and forecast non-stationary multivariate time series with integrated differencing and moving average components.

### State Space & Kalman Filtering
- **State Space Models:** Build and update state space representations for time series and macro models.
- **Kalman Filter & Smoother:** Perform filtering, smoothing, and likelihood evaluation for latent state estimation.

### Deep Learning for Time Series
- **N-BEATS:** Modern neural network architecture for interpretable, high-accuracy time series forecasting.

### Advanced Tools
- **Optimization Algorithms:**
  - Random Walk Metropolis (RWM) for Bayesian estimation (MCMC)
  - Genetic Algorithms for global optimization
  - Simulated Annealing for robust parameter search
- **Priors:** Specify and customize prior distributions for Bayesian estimation.
- **State-Space Updates:** Update state-space solutions for DSGE and time series models.

---

## Installation

Install Econometron via pip:

```bash
pip install econometron
```

Or clone/download the latest version from the [GitHub repository](https://github.com/Amineouerfelli/econometron):

```bash
git clone https://github.com/Amineouerfelli/econometron.git
git install ./econometron
```

---

## Example Usage

### 1. VAR Model
```python
from econometron.Models.Base_Var import VAR
import pandas as pd
# df: DataFrame with columns ['date', 'interest', 'inflation', 'gdp']
var_model = VAR(max_p=4, criterion='AIC', forecast_horizon=8, plot=True)
var_model.fit(df, date_column='date', columns=['interest', 'inflation', 'gdp'])
forecasts = var_model.predict()
```

### 2. DSGE Model (Linear Rational Expectations)
```python
from econometron.Models.Linear_RE import RE_model
model = RE_model(equations, variables, exo_states, endo_states, parameters)
solution = model.solve()
```

### 3. N-BEATS Time Series Forecasting
```python
from econometron.Models.n_beats import NBeatsModel, create_sliding_windows, normalize_data
import scipy.io
# Load data
data = scipy.io.loadmat('Z.mat')["Z"][0, :]
X, y = create_sliding_windows(data, input_size=100, forecast_horizon=20)
X_norm, y_norm, mean, std = normalize_data(X, y)
model = NBeatsModel(input_size=100, hidden_size=512, forecast_horizon=20, stack_configs=[...])
model.fit(X_norm, y_norm, epochs=100, batch_size=32)
```

### 4. State Space & Kalman Filter

---

## 📚 Documentation & Notebooks
- See econometron.netlify.app with full and example notebooks for detailed usage and examples.

---

## 🤝 Contributing
Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

---

## 📄 License
MIT License

---

## 📬 Contact
For questions or support, contact :[mohamedamine.ouerfelli@outlook.com](mailto:mohamedamine.ouerfelli@outlook.com)
