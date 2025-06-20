# 🧮 econometron.filters — API Guide & Mathematical Walkthrough

---

> **The `econometron.filters` module provides robust tools for filtering and smoothing in state-space models, focusing on Kalman filtering and smoothing for economic and time series applications.**

---

## ✨ Key Features

- **Kalman Filter & Smoother:** Forward and backward passes for optimal state estimation.
- **State-Space Modeling:** Linear Gaussian models for dynamic systems.
- **Parameter Estimation:** Log-likelihood computation for MLE.
- **Flexible API:** Class-based and functional interfaces.
- **Mathematical Rigor:** Clear equations and notation for transparency.

---

## Mathematical Foundation

### State-Space Model

**State (Transition) Equation:**
$$
x_t = A x_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, Q)
$$

**Observation (Measurement) Equation:**
$$
y_t = D x_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, R)
$$

**Where:**
- $x_t$: Hidden state vector ($n \times 1$)
- $y_t$: Observed vector ($m \times 1$)
- $A$: State transition matrix ($n \times n$)
- $D$: Observation matrix ($m \times n$)
- $Q$: Process noise covariance ($n \times n$)
- $R$: Observation noise covariance ($m \times m$)

---

## Kalman Filter (Forward Pass)

**Prediction Step:**
$$
\hat{x}_{t|t-1} = A \hat{x}_{t-1|t-1}
$$
$$
P_{t|t-1} = A P_{t-1|t-1} A^\top + Q
$$

**Update Step:**
$$
K_t = P_{t|t-1} D^\top (D P_{t|t-1} D^\top + R)^{-1}
$$
$$
\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - D \hat{x}_{t|t-1})
$$
$$
P_{t|t} = (I - K_t D) P_{t|t-1}
$$

---

## Kalman Smoother (Backward Pass)

**Smoothing Step:**
$$
C_t = P_{t|t} A^\top P_{t+1|t}^{-1}
$$
$$
\hat{x}_{t|T} = \hat{x}_{t|t} + C_t (\hat{x}_{t+1|T} - \hat{x}_{t+1|t})
$$
$$
P_{t|T} = P_{t|t} + C_t (P_{t+1|T} - P_{t+1|t}) C_t^\top
$$

---

## Log-Likelihood for Parameter Estimation

$$
\log L = -\frac{1}{2} \sum_{t=1}^{T} \left( m \log 2\pi + \log |S_t| + v_t^\top S_t^{-1} v_t \right)
$$

Where:
- $v_t = y_t - D \hat{x}_{t|t-1}$: Innovation (residual)
- $S_t = D P_{t|t-1} D^\top + R$: Innovation covariance

---

## API Reference

### Functions
- **`kal_smooth(...)`** — Executes the Kalman smoother given observations and model parameters.
- **`kalman_objective(...)`** — Computes the negative log-likelihood for parameter estimation.

### Class: `Kalman`
Encapsulates the Kalman filter and smoother logic.

**Constructor:**
- `A` (ndarray): State transition matrix $(n \times n)$
- `D` (ndarray): Observation matrix $(m \times n)$
- `Q` (ndarray): Process noise covariance $(n \times n)$
- `R` (ndarray): Observation noise covariance $(m \times m)$
- `x0` (ndarray, optional): Initial state mean $(n,)$, defaults to zero vector
- `P0` (ndarray, optional): Initial covariance $(n \times n)$, defaults to $10^6 \cdot I_n$

**Methods:**
- `filter(y)`: Runs the Kalman filter on observed data. Returns filtered states, covariances, and log-likelihood.
- `smooth(y)`: Runs the Kalman smoother on observed data. Returns smoothed states and covariances.

---

## Notes

- The module is tailored for economic modeling but is general for any linear Gaussian state-space model.
- For interactive use, consider converting this guide to a Jupyter Notebook or PDF.

---

## References

- Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

---

## License
MIT License.

---

## Contact
Contact the Econometron team: [mohamedamine.ouerfelli@outlook.com](mailto:mohamedamine.ouerfelli@outlook.com)

---

## Example

```python
from econometron.filters import Kalman_class
kf = Kalman_class(...)
kf.filter()
```