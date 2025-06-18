# Econometron

Econometron is a Python module for approximating, solving, and simulating dynamic stochastic general equilibrium (DSGE) models. It provides advanced tools for both classical and modern time series analysis, with a strong focus on rational expectations modeling and macroeconomic simulation.

## Core Model Types

- **Vector Autoregression (VAR):**
  Multivariate time series model for analyzing the dynamic impact and interdependencies among several time series. Implements OLS estimation, lag selection, and forecasting.

- **Vector Autoregressive Integrated Moving Average (VARIMA):**
  Extends VAR to handle non-stationary data by including differencing and moving average components. Suitable for multivariate time series with trends or seasonality.

- **Linear Rational Expectations (Linear RE):**
  Framework for solving and simulating linear rational expectations models, commonly used in macroeconomic DSGE analysis. Handles symbolic equations, state/control variable separation, and solution via linear algebra.

- **Nonlinear Rational Expectations (Nonlinear RE):**
  Abstract base for nonlinear rational expectations models. Designed for custom solution methods where model equations are nonlinear in states or controls.

- **N-BEATS (Neural Basis Expansion for Time Series):**
  Deep learning model for time series forecasting. Implements block, stack, and full model architectures with customizable basis functions (e.g., Chebyshev polynomials). Suitable for both univariate and multivariate forecasting tasks.

## Features

- Kalman filter and smoother algorithms
- Bayesian and Maximum Likelihood estimation
- Utilities for time series data preparation and processing
- Optimizers and solvers for parameter estimation
- Extensible architecture for custom econometric and DSGE models

## Project Structure

- `econometron/`: Main package source code
  - `filters/`: Kalman filter, smoother, and related algorithms
  - `Models/`: Model classes (VAR, ARIMA, Rational Expectations, etc.)
  - `utils/`: Utilities for data preparation, estimation, optimization, and state space modeling
- `test/`: Unit tests and example notebooks
- `plots/`: Output plots and visualizations
- `requirements.txt`: Python dependencies
- `setup.py`: Installation script

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Install the package:
   ```
   pip install .
   ```
3. Explore example notebooks in the `test/` directory.

## License

This project is licensed under the MIT License.
