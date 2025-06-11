# Econometron

**Econometron** is a Python library designed for building, simulating, and estimating macroeconomic models, with a focus on **Dynamic Stochastic General Equilibrium (DSGE)** , **Vector Autoregression (VAR)** , **Vector Autoregression Integrated Moving Average (VARIMA)** models. It offers a modular architecture and a suite of tools tailored for economists, researchers, and data scientists working in quantitative macroeconomics.

## Features

### DSGE Modelling
Econometron provides a robust framework for working with DSGE models:
- **Model Definition**: Specify model equations, parameters, and variables (e.g., consumption, capital, labor) in a flexible, user-friendly syntax.
- **Simulation**: Generate time series data under stochastic shocks, such as technology or monetary policy shocks.
- **Estimation**: Calibrate and estimate model parameters using numerical solvers and optimization techniques.
- **Numerical Solvers**: Solve nonlinear or linearized DSGE models efficiently.

### VAR Modelling
Econometron supports Vector Autoregression (VAR) models for analyzing multivariate time series:
- **Model Specification**: Define VAR models with multiple time series (e.g., GDP, inflation, interest rates).
- **Estimation**: Estimate VAR parameters using Ordinary Least Square.
- **Forecasting**: Generate forecasts and impulse response functions.

### Advanced Tools
- **Kalman Filter & Smoother**:perform filtering and Smoothing for state estimation and likelihood evaluation.
- **Optimization Algorithms**:
  - **Random Walk Metropolis (RWM)**: Bayesian estimation via Markov Chain Monte Carlo (MCMC) methods.
  - **Genetic Algorithms**: Global optimization for complex parameter spaces.
  - **Simulated Annealing**: Robust optimization for non-smooth or nonlinear problems.
- **Priors**: Specify and customize prior distributions for Bayesian estimation.
- **State-Space Updates**: Update state-space solution for DSGE models.

## Installation
To install Econometron, use pip:

```bash
pip install econometron 
```
