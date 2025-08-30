
![Econometron Logo](/econometron.png)

# Econometron: A Python package for Econometric and Time Series Analysis

## Introduction

Econometron is a Python library designed for econometric modeling, time series analysis, and dynamic stochastic general equilibrium (DSGE) model solving and estimation. It provides a wide range of tools for researchers, economists, and data scientists to build, estimate, and analyze complex multivariate time series models and non-linear DSGE models. With a focus on flexibility and performance, Econometron supports both classical and modern approaches, including state-of-the-art neural network-based forecasting and robust statistical methods.

Whether you're modeling economic time series, performing impulse response function (IRF) analysis, or solving non-linear DSGE models, Econometron offers a unified and efficient framework to streamline your workflow.

## Key Features

### Multivariate Statistical Time Series Models
- **VAR (Vector Autoregression):** Model the dynamic relationships between multiple time series.  
- **SVAR (Structural Vector Autoregression):** Incorporate structural restrictions for causal inference and policy analysis.  
- **VARMA (Vector Autoregressive Moving Average):** Combine autoregressive and moving average components for enhanced flexibility.  
- **VARIMA (Vector Autoregressive Integrated Moving Average):** Handle non-stationary time series with differencing.  

### VARMA Identification
- **Echelon Form Identification:** Implements the echelon form approach for identifying VARMA models, ensuring robust and unique parameter estimation in Python.

### Neural Network-Based Forecasting
- **N-BEATS (Neural Basis Expansion Analysis for Time Series):** A state-of-the-art deep learning model for univariate and multivariate time series forecasting.  
- **N-BEATS + RevIN:** Enhances N-BEATS with Reversible Instance Normalization (RevIN) for improved generalization and robustness.

### State Space Models
- Flexible framework for modeling complex dynamic systems using state space representations, suitable for both linear and non-linear systems.

### Estimation Methods
- **Bayesian Estimation:** Leverage Bayesian techniques for parameter estimation, incorporating prior knowledge and uncertainty quantification.  
- **Maximum Likelihood Estimation (MLE):** Optimize model parameters using likelihood-based methods for precise inference.

### Impulse Response Functions (IRF)
- **Local Projection IRF:** Compute impulse response functions using local projection methods, ideal for non-linear and robust analysis.

### Non-Linear DSGE Model Solving
- **Projection Methods:** Solve non-linear DSGE models using advanced numerical techniques:  
  - Galerkin Method: Project solutions onto a basis of functions for accurate approximation.  
  - Collocation Method: Solve at specific points to approximate the policy function.  
  - Least Squares Method: Minimize residuals to find optimal solutions.


## Getting Started
To use Econometron, install it via pip:

```bash
pip install econometron
````

### Example: Fitting a VAR Model

```python
from econometron.Models.VectorAutoReg import VAR

# Load your time series data
data = ...  # Your multivariate time series data
model = VAR(data=data,max_p=2,check_stationnarity=True)
results = model.fit()
```

For detailed documentation, tutorials, and examples, visit the [Econometron Documentation](https://econometron.netlify.app).



## Why Econometron?

* **Comprehensive:** Covers a wide range of econometric models, from classical VAR to cutting-edge neural network approaches.
* **Flexible:** Supports both statistical and machine learning-based methods for time series analysis.
* **Robust:** Implements state-of-the-art estimation and identification techniques for reliable results.
* **User-Friendly:** Designed with Python's ecosystem in mind, integrating seamlessly with libraries like NumPy, Pandas, and PyTorch.



## Code of Conduct

We are committed to fostering a welcoming and inclusive community. All participants are expected to:

* Be respectful and considerate in interactions.
* Avoid harassment or discriminatory behavior.
* Use constructive feedback and maintain professionalism.
* Respect the community and project’s guidelines.

Violations may result in removal from the project or community channels. Please read the full [Code of Conduct](/CODE_OF_CONDUCT.md) for details.



## Contributing

Contributions are highly valued. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes with clear, well-documented code.
4. Run all tests to ensure stability.
5. Submit a pull request describing your changes in detail.

For more information, see the full [Contributing Guide](/CONTRIBUTING.md).



## License

Econometron is licensed under the MIT License. See the LICENSE file for more information.
