import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from econometron.Models.VectorAutoReg import VARMA 
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, breaks_cusumolsresid
from sklearn.cross_decomposition import CCA
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

# Fixture to create a VARMA instance
@pytest.fixture
def varma_model(sample_data):
    return VARMA(data=sample_data, max_p=2, max_q=2, columns=['var1', 'var2'], criterion='AIC',
                 forecast_h=10, plot=False, bootstrap_n=100, ci_alpha=0.05, check_stationarity=False,
                 structural_id=False, enforce_stab_inver=False)

# Test initialization
def test_varma_initialization(sample_data):
    varma = VARMA(data=sample_data, max_p=2, max_q=2, columns=['var1', 'var2'], criterion='AIC',
                  forecast_h=10, bootstrap_n=100, ci_alpha=0.05, check_stationarity=False,
                  structural_id=True, enforce_stab_inver=True, Threshold=0.8)
    assert isinstance(varma.data, pd.DataFrame)
    assert varma.max_p == 2
    assert varma.max_q == 2
    assert varma.criterion == 'AIC'
    assert varma.forecast_h == 10
    assert varma.ci_alpha == 0.05
    assert varma.Threshold == 0.8
    assert varma.structural_id
    assert varma.stab_inver
    assert not varma.fitted
    assert varma.Kronind is None
    assert varma.best_model == {}

# Test inherited data validation (from VAR)
def test_validate_data_inherited(sample_data, varma_model):
    varma_model._validate_the_data(sample_data, verbose=False)
    assert varma_model.data.equals(sample_data)
    assert not varma_model.stationarity_results

def test_validate_data_invalid_type():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="The input data must be a pandas DataFrame"):
        varma = VARMA(data=data, check_stationarity=False)
        varma._validate_the_data(data)

# Test Kronecker index computation
@patch('statsmodels.tsa.stattools.acf')
@patch('sklearn.cross_decomposition.CCA.fit_transform')
def test_kron_index(mock_cca, mock_acf, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_acf.return_value = np.ones(10) * 0.1  # Mock ACF to avoid issues
    mock_cca.return_value = (np.random.randn(98, 2), np.random.randn(98, 2))  # Mock CCA outputs
    result = varma_model.kron_index(lag=2)
    assert 'index' in result
    assert 'tests' in result
    assert result['index'].shape == (2,)
    assert len(result['tests']) > 0

# Test structural identification
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
def test_struct_id(mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    varma_model.structural_id = True
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((5, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((5, 2)), 'z_values': np.zeros((5, 2)), 'p_values': np.ones((5, 2))})
    result = varma_model.struct_id(ord=2, output=False)
    assert 'AR_s_id' in result
    assert 'MA_s_id' in result
    assert result['AR_s_id'].shape == (2, 4) 
    assert result['MA_s_id'].shape == (2, 4)
    assert varma_model.best_p == 1
    assert varma_model.best_q == 1

# Test initial estimation (_ini_s1)
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
def test_ini_s1(mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    varma_model.structural_id = False
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    estimates, stand_err = varma_model._ini_s1(p=2, q=2, output=False)
    assert estimates.shape == (9, 2) 
    assert stand_err.shape == (9, 2)

# Test parameter preparation
def test_prepare_for_est(varma_model):
    estimates = np.ones((8, 2))
    stand_err = np.ones((8, 2)) * 0.1
    par, separ, lb, ub = varma_model._prepare_for_est(estimates, stand_err, output=False)
    assert par.shape == (16,)  # Flattened
    assert separ.shape == (16,)
    assert lb.shape == (16,)
    assert ub.shape == (16,)
    assert np.allclose(lb, par - 2 * separ)
    assert np.allclose(ub, par + 2 * separ)

# Test AR and MA matrix preparation
def test_prepare_A_B_matrices(varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    par = np.ones((10)) 
    A, B, Cst, indexes = varma_model.prepare_A_B_Matrices(par, p=1, q=1)
    assert A.shape == (1, 2, 2)  
    assert B.shape == (1, 2, 2)  
    assert Cst.shape == (1, 2) 
    assert len(indexes) == 10

# Test log-likelihood function
def test_LL_func(varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    par = np.zeros((10))
    neg_ll = varma_model.LL_func(par, p=1, q=1, verbose=False)
    assert isinstance(neg_ll, float)
    assert not np.isnan(neg_ll)

# Test companion matrix
def test_build_companion(varma_model):
    K = 2
    matrix = np.ones((2, K, K))
    companion = varma_model._build_companion(matrix, K, order=2)
    assert companion.shape == (4, 4)
    assert np.all(companion[:K, :K] == matrix[0])
    assert np.all(companion[:K, K:2*K] == matrix[1])
    assert np.all(companion[K:, :-K] == np.eye(2))

# Test diagnostics computation
def test_compute_diags(varma_model):
    model = {
        'par': np.ones((10)),
        'p': 1,
        'q': 1
    }
    varma_model.best_model = model
    diags = varma_model.compute_diags(gs=True, model=model)
    assert 'se' in diags
    assert 'tvals' in diags
    assert 'pvals' in diags
    assert 'signif' in diags
    assert diags['se'].shape == (10,)
    assert diags['tvals'].shape == (10,)
    assert diags['pvals'].shape == (10,)
    assert len(diags['signif']) == 10

# Test fit method (non-structural)
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
def test_fit_non_structural(mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    result = varma_model.fit(p=2, q=2, plot=False, verbose=False)
    print(result)
    assert varma_model.fitted == False
    assert 'par' in result
    assert 'A' in result
    assert 'B' in result
    assert 'Cst' in result
    assert 'residuals' in result
    assert 'loglikelihood' in result

# Test fit method (grid search)
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
def test_fit_grid_search(mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    result = varma_model.fit(p=None, q=None, plot=False, verbose=False)
    assert varma_model.fitted == False
    assert varma_model.best_p in range(varma_model.max_p + 1)
    assert varma_model.best_q in range(varma_model.max_q + 1)
    assert 'par' in result
    assert 'A' in result
    assert 'B' in result

# Test diagnostics
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
@patch('statsmodels.stats.diagnostic.acorr_ljungbox')
@patch('statsmodels.stats.diagnostic.het_arch')
@patch('statsmodels.stats.diagnostic.breaks_cusumolsresid')
def test_run_full_diagnosis(mock_cusum, mock_arch, mock_ljungbox, mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    mock_ljungbox.return_value = pd.DataFrame({'lb_pvalue': [0.1]})
    mock_arch.return_value = (0.0, 0.1, 0.0, 0.0)
    mock_cusum.return_value = (0.0, 0.1, [(5, 1.36)])
    varma_model.fit(p=2, q=2, plot=False, verbose=False)
    diagnosis = varma_model.run_full_diagnosis(plot=False, threshold=0.8)
    assert diagnosis['Final_score'] >= 0
    assert diagnosis['Verdict'] in ['Passed', 'Failed']
    assert 'Stability Score' in diagnosis
    assert 'Invertibility Score' in diagnosis
    assert 'Autocorrelation_score' in diagnosis
    assert 'Normality_score' in diagnosis
    assert 'Structural_breaks_score' in diagnosis

# Test prediction
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
def test_predict(mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    varma_model.fit(p=2, q=2, plot=False, verbose=False)
    result = varma_model.predict(n_periods=10, plot=False)
    assert result['point'].shape == (10, 2)
    assert result['ci_lower'].shape == (10, 2)
    assert result['ci_upper'].shape == (10, 2)
    assert isinstance(result['point'], pd.DataFrame)

# Test simulation
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
def test_simulate(mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    varma_model.fit(p=2, q=2, plot=False, verbose=False)
    result = varma_model.simulate(n_periods=100, plot=False)
    assert result['simulations'].shape == (100, 2)

# Test impulse response
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
def test_impulse_res(mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    varma_model.fit(p=2, q=2, plot=False, verbose=False)
    result = varma_model.impulse_res(h=10, bootstrap=False, plot=False)
    assert result['irf'].shape == (11, 2, 2)

@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
def test_impulse_res_bootstrap(mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    varma_model.fit(p=2, q=2, plot=False, verbose=False)
    result = varma_model.impulse_res(h=10, bootstrap=True, n_boot=10, plot=False)
    assert result['irf'].shape == (11, 2, 2)
    assert result['ci_lower'].shape == (11, 2, 2)
    assert result['ci_upper'].shape == (11, 2, 2)

# Test FEVD
@patch('econometron.utils.estimation.Regression.ols_estimator')
@patch('econometron.Models.VectorAutoReg.VAR.fit')
@patch('matplotlib.pyplot.show')
def test_fevd(mock_show, mock_var_fit, mock_ols, varma_model, sample_data):
    varma_model.data = sample_data
    varma_model.columns = ['var1', 'var2']
    mock_var_fit.return_value = {'p': 2, 'beta': np.zeros((5, 2)), 'residuals': np.random.randn(98, 2)}
    mock_ols.return_value = (np.zeros((8, 2)), np.zeros((98, 2)), np.zeros((98, 2)),
                             {'se': np.ones((8, 2)), 'z_values': np.zeros((8, 2)), 'p_values': np.ones((8, 2))})
    varma_model.fit(p=2, q=2, plot=False, verbose=False)
    fevd = varma_model.FEVD(h=10, plot=False)
    assert fevd.shape == (10, 2, 2)
    for i in range(10):
        for j in range(2):
            assert np.allclose(np.sum(fevd[i, j, :]), 1.0, atol=1e-6)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])