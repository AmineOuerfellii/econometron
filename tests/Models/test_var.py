import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from econometron.Models.VectorAutoReg import VAR 
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixture to create synthetic multivariate time series data
@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    var1 = np.sin(0.1 * t) + 0.1 * np.random.randn(n)
    var2 = np.cos(0.1 * t) + 0.1 * np.random.randn(n)
    data = pd.DataFrame({'var1': var1, 'var2': var2}, index=pd.date_range('2020-01-01', periods=n))
    return data

# Fixture to create a VAR instance
@pytest.fixture
def var_model(sample_data):
    return VAR(data=sample_data, max_p=2, columns=['var1', 'var2'], criterion='AIC', 
               forecast_horizon=10, plot=False, bootstrap_n=100, ci_alpha=0.05, 
               check_stationarity=False, verbose=False)

# Test initialization
def test_var_initialization(sample_data):
    var = VAR(data=sample_data, max_p=2, columns=['var1', 'var2'], criterion='AIC',
              forecast_horizon=10, bootstrap_n=100, ci_alpha=0.05, check_stationarity=False)
    assert isinstance(var.data, pd.DataFrame)
    assert var.max_p == 2
    assert var.criterion == 'AIC'
    assert var.forecast_horizon == 10
    assert var.ci_alpha == 0.05
    assert var.thershold == 0.8
    assert not var.check_stationarity
    assert not var.fitted

def test_var_invalid_ci_alpha(sample_data):
    var = VAR(data=sample_data, max_p=2, ci_alpha=1.5, check_stationarity=False)
    assert var.ci_alpha == 0.05  # Should fallback to default
    var = VAR(data=sample_data, max_p=2, ci_alpha=0, check_stationarity=False)
    assert var.ci_alpha == 0.05  # Should fallback to default

def test_var_invalid_threshold(sample_data):
    var = VAR(data=sample_data, max_p=2, Threshold=1.5, check_stationarity=False)
    assert var.thershold == 0.8  # Should fallback to default
    var = VAR(data=sample_data, max_p=2, Threshold=0, check_stationarity=False)
    assert var.thershold == 0.8  # Should fallback to default

# Test data validation
def test_validate_data_valid(sample_data, var_model):
    var_model._validate_the_data(sample_data, verbose=False)
    assert var_model.data.equals(sample_data)
    assert not var_model.stationarity_results

def test_validate_data_invalid_type():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="The input data must be a pandas DataFrame"):
        var = VAR(data=data, check_stationarity=False)
        var._validate_the_data(data)

def test_validate_data_nan():
    data = pd.DataFrame({'var1': [1, np.nan, 3], 'var2': [4, 5, 6]})
    with pytest.raises(ValueError, match="Columns is entirely or contains NaN values"):
        var = VAR(data=data, check_stationarity=False)
        var._validate_the_data(data)

# Test stationarity checks (mock ADF and KPSS to avoid long computation)
@patch('statsmodels.tsa.stattools.adfuller')
@patch('statsmodels.tsa.stattools.kpss')
def test_validate_data_stationarity(mock_kpss, mock_adf, sample_data):
    mock_adf.return_value = (0.0, 0.01, None, None, {'5%': -2.86})
    mock_kpss.return_value = (0.0, 0.1, None, {'5%': 0.463})
    var = VAR(data=sample_data, check_stationarity=True, verbose=False)
    var._validate_the_data(sample_data, verbose=False)
    assert var.stationarity_results['var1']
    assert var.stationarity_results['var2']
# Test lag matrix
def test_lag_matrix(var_model, sample_data):
    var_model.data = sample_data
    X, Y = var_model.lag_matrix(2)
    assert X.shape == (98, 4)  # T - lags, K * lags
    assert Y.shape == (98, 2)  # T - lags, K
    assert np.allclose(Y, sample_data.values[2:])

def test_lag_matrix_too_many_lags(var_model, sample_data):
    var_model.data = sample_data
    with pytest.raises(ValueError, match="lags are superior to the series length"):
        var_model.lag_matrix(lags=100)

# Test order selection
@patch('econometron.utils.estimation.Regression.ols_estimator')
def test_order_select(mock_ols, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    table = var_model.order_select()
    assert table.shape == (2, 4)  # p, aic, bic, hqic
    assert list(table['p']) == [2, 1]
    assert var_model.criterion.lower() in ['aic', 'bic', 'hqic']

# Test fit method
@patch('econometron.utils.estimation.Regression.ols_estimator')
def test_fit(mock_ols, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2)), 'log_likelihood': -100, 'R2': 0.8})
    result = var_model.fit(columns=['var1', 'var2'])
    assert var_model.fitted
    assert var_model.best_p in [1, 2]
    assert var_model.best_model is not None
    assert var_model.coeff_table.shape == (5, 8)  # 1 constant + 2 lags * 2 vars, 4 stats per var

def test_fit_invalid_columns(var_model, sample_data):
    var_model.data = sample_data
    with pytest.raises(ValueError, match="Some of The Columns don't exist"):
        var_model.fit(columns=['var1', 'var3'])

def test_fit_insufficient_data(var_model):
    short_data = pd.DataFrame({'var1': [1, 2], 'var2': [3, 4]})
    var_model.data = short_data
    with pytest.raises(ValueError, match="Insufficient observations"):
        var_model.fit(columns=['var1', 'var2'])

# Test diagnostics
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
@patch('statsmodels.stats.diagnostic.acorr_ljungbox')
@patch('statsmodels.stats.diagnostic.het_arch')
@patch('statsmodels.stats.diagnostic.breaks_cusumolsresid')
def test_run_full_diagnosis(mock_cusum, mock_arch, mock_ljungbox, mock_ols, mock_show, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2)), 'log_likelihood': -100, 'R2': 0.8})
    mock_ljungbox.return_value = pd.DataFrame({'lb_pvalue': [0.1]})
    mock_arch.return_value = (0.0, 0.1, 0.0, 0.0)
    mock_cusum.return_value = (0.0, 0.1, [(5, 1.36)])
    var_model.fit(columns=['var1', 'var2'])
    diagnosis = var_model.run_full_diagnosis(plot=False, threshold=0.8)
    assert diagnosis['Final_score'] >= 0
    assert diagnosis['Verdict'] in ['Passed', 'Failed']
    assert 'Stability Score' in diagnosis
    assert 'Autocorrelation_score' in diagnosis
    assert 'Normality_score' in diagnosis

# Test prediction
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
def test_predict(mock_show, mock_ols, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    var_model.fit(columns=['var1', 'var2'])
    result = var_model.predict(n_periods=10, plot=False)
    assert result['point'].shape == (10, 2)
    assert result['ci_lower'].shape == (10, 2)
    assert result['ci_upper'].shape == (10, 2)
    assert isinstance(result['point'], pd.DataFrame)

# Test impulse response
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
def test_impulse_res(mock_show, mock_ols, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    var_model.fit(columns=['var1', 'var2'])
    irf = var_model.impulse_res(h=10, orth=True, bootstrap=False, plot=False)
    assert irf.shape == (11, 2, 2)

@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
def test_impulse_res_bootstrap(mock_show, mock_ols, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    var_model.fit(columns=['var1', 'var2'])
    result = var_model.impulse_res(h=10, orth=True, bootstrap=True, n_boot=10, plot=False)
    assert result['irf'].shape == (11, 2, 2)
    assert result['ci_lower'].shape == (11, 2, 2)
    assert result['ci_upper'].shape == (11, 2, 2)

# Test FEVD
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
def test_fevd(mock_show, mock_ols, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    var_model.fit(columns=['var1', 'var2'])
    fevd = var_model.FEVD(h=10, plot=False)
    assert fevd.shape == (11, 2, 2)
    for i in range(11):
        for j in range(2):
            assert np.allclose(np.sum(fevd[i, j, :]), 1.0, atol=1e-6)

# Test simulation
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
def test_simulate(mock_show, mock_ols, var_model, sample_data):
    var_model.data = sample_data
    var_model.columns = ['var1', 'var2']
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    var_model.fit(columns=['var1', 'var2'])
    result = var_model.simulate(n_periods=100, plot=False)
    assert result['simulations'].shape == (100, 2)


@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
def test_el_rapido_mode(mock_show, mock_ols, sample_data):
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    var = VAR(data=sample_data, max_p=2, Key='EL_RAPIDO', check_stationarity=False, plot=False)
    assert var.fitted
    assert var.best_model is not None


@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('matplotlib.pyplot.show')
@patch('builtins.input', side_effect=['AIC', 'y', 'y', '1', 'y', 'y', 'y', 'y'])
def test_sbs_mode(mock_input, mock_show, mock_ols, sample_data):
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)), 
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 
                              'p_values': np.ones((5, 2))})
    var = VAR(data=sample_data, max_p=2, Key='SbS', check_stationarity=False, plot=False)
    assert var.fitted
    assert var.best_p == 2
    assert var.best_model is not None

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])