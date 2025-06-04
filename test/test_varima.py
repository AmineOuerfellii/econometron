import numpy as np
import pandas as pd
import pytest
from econometron.Models.varima import VARIMA

def generate_synthetic_varima_data(n_obs=100, n_vars=2, ar_params=None, ma_params=None, seed=42):
    np.random.seed(seed)
    if ar_params is None:
        ar_params = [np.array([[0.5, 0.1], [0.0, 0.4]])]
    if ma_params is None:
        ma_params = [np.array([[0.2, 0.0], [0.0, 0.1]])]
    K = n_vars
    max_p = len(ar_params)
    max_q = len(ma_params)
    y = np.zeros((n_obs + max_p, K))
    e = np.random.normal(size=(n_obs + max_q, K))
    for t in range(max_p, n_obs + max_p):
        ar_sum = sum(ar_params[lag] @ y[t - lag - 1] for lag in range(max_p))
        ma_sum = sum(ma_params[lag] @ e[t - lag - 1] for lag in range(max_q))
        y[t] = ar_sum + ma_sum + e[t]
    dates = pd.date_range(start='2000-01-01', periods=n_obs, freq='M')
    df = pd.DataFrame(y[max_p:], columns=[f'y{i+1}' for i in range(K)], index=dates)
    return df

def test_varima_fit_and_predict():
    data = generate_synthetic_varima_data(n_obs=60, n_vars=2)
    model = VARIMA(max_p=2, max_q=1, criterion='AIC', max_diff=0, forecast_horizon=4, plot=False)
    fitted = model.fit(data)
    assert model.fitted
    assert model.best_p is not None and model.best_q is not None
    assert hasattr(model, 'best_model') and model.best_model is not None
    # Forecast
    forecast_df = model.predict(h=4)
    assert forecast_df.shape[0] == 4
    for col in data.columns:
        assert col in forecast_df.columns
        assert f'{col}_ci_lower' in forecast_df.columns
        assert f'{col}_ci_upper' in forecast_df.columns
    # Check forecast values are finite
    assert np.all(np.isfinite(forecast_df.values))

def test_varima_edge_case_short_series():
    data = generate_synthetic_varima_data(n_obs=10, n_vars=2)
    model = VARIMA(max_p=1, max_q=1, criterion='AIC', max_diff=0, forecast_horizon=2, plot=False)
    with pytest.raises(ValueError):
        model.fit(data)

def test_varima_stationarity_diff():
    # Non-stationary data (random walk)
    np.random.seed(0)
    n_obs = 50
    y = np.cumsum(np.random.normal(size=(n_obs, 2)), axis=0)
    dates = pd.date_range(start='2010-01-01', periods=n_obs, freq='M')
    df = pd.DataFrame(y, columns=['y1', 'y2'], index=dates)
    model = VARIMA(max_p=1, max_q=0, criterion='AIC', max_diff=1, forecast_horizon=2, plot=False)
    fitted = model.fit(df)
    assert model.fitted
    # Should have differenced at least one variable
    assert any(v > 0 for v in model.diff_orders.values())
