# **Econometron**

**Econometron** is a Python module designed for approximating, solving, and simulating **Dynamic Stochastic General Equilibrium (DSGE)** models. It combines classical econometric techniques with modern machine learning tools to offer a flexible framework for macroeconomic modeling, time series forecasting, and rational expectations analysis.
It currently supports : VAR ,VARIMA , N-Beats Models
---

## **Core Model Types**

* **Vector Autoregression (VAR):**
  A multivariate model that captures the interdependencies and dynamic relationships among multiple time series. Includes OLS estimation, optimal lag selection, and forecasting capabilities.

* **Vector Autoregressive Integrated Moving Average (VARIMA):**
  An extension of VAR for non-stationary time series, incorporating differencing and moving average components. Ideal for modeling trending or seasonal multivariate data.

* **Linear Rational Expectations (Linear RE):**
  A solver for linear rational expectations models often used in DSGE frameworks. Supports symbolic modeling, state/control variable classification, and linear algebra-based solutions.

* **Nonlinear DSGE (Nonlinear RE):**
  An abstract framework for implementing nonlinear DSGE models. Provides the foundation for defining and solving models with nonlinearities in states or controls.

* **N-BEATS (Neural Basis Expansion for Time Series):**
  A state-of-the-art deep learning model for time series forecasting. Supports flexible architecture configurations (block, stack, full) and customizable basis functions like Chebyshev polynomials. Works with both univariate and multivariate data.

---

## **Key Features**

* Kalman Filter and Smoother implementations
* Bayesian and Maximum Likelihood estimation methods
* Tools for time series data preparation and transformation
* Optimizers and solvers for structural and reduced-form models
* Modular and extensible architecture for building custom models

---

## **Project Structure**

```
econometron/
├── filters/         # Kalman filtering and smoothing algorithms
├── models/          # Core model classes (VAR, ARIMA, RE, etc.)
├── utils/           # Data handling, estimation, and optimization tools
test/                # Unit tests and example notebooks
plots/               # Visual outputs and diagnostics
requirements.txt     # Project dependencies
setup.py             # Installation script
```

---

## **Getting Started**

1. **Download the source**
   [Download here](https://github.com/Amineouerfellii/econometron)

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**

   ```bash
   python setup.py install
   ```

---

## **Documentation**

* **Econometron Docs:**
  [https://econometron.netlify.app]https://(https://econometron.netlify.app))

---

## **License**

This project is licensed under the **MIT License** — free for personal and commercial use, with proper attribution.

---

Let me know if you'd like help writing an [example notebook](f), [API reference](f), or [contribution guide](f).
