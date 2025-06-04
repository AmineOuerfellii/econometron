import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import econometron
from econometron.Models import VAR
import pytest



def test_var_fit_and_predict():
    # Create sample multivariate time series
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
    np.random.seed(42)
    n = len(dates)
    series1 = np.random.randn(n) + np.linspace(0, 5, n)
    series2 = np.sin(np.linspace(0, 10, n)) + np.random.randn(n) * 0.5
    data = pd.DataFrame({
        'series1': series1,
        'series2': series2,
        'date': dates
    })
    
    # Introduce missing values
    data.loc[data.sample(frac=0.1).index, 'series1'] = np.nan
    
    # Initialize and fit VAR model
    var_model = VAR(max_p=9, criterion='AIC', forecast_horizon=5, plot=False)
    var_model.fit(data, date_column='date', columns=['series1', 'series2'])
    
    # Generate forecasts with confidence intervals
    forecasts = var_model.predict()
    
    assert forecasts is not None
    assert isinstance(forecasts, pd.DataFrame)
    assert not forecasts.empty

    print("\nForecasts with Confidence Intervals:")
    print(forecasts.round(4))