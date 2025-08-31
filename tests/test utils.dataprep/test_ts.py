import pytest
import pandas as pd
import numpy as np
from econometron.utils.data_preparation import TransformTS 

# Sample data for tests
@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    var1 = np.sin(0.1 * t) + 0.1 * np.random.randn(n)
    var2 = np.cos(0.1 * t) + 0.1 * np.random.randn(n)
    data = pd.DataFrame({'var1': var1, 'var2': var2}, index=pd.date_range('2020-01-01', periods=n))
    return data

def test_initialization(sample_data):
    ts = TransformTS(sample_data)
    # Check transformed_data exists
    assert ts.transformed_data is not None
    # Columns should match numeric columns
    assert ts.columns == sample_data.select_dtypes(include=np.number).columns.tolist()

def test_diff_transformation(sample_data):
    ts = TransformTS(sample_data, method='diff', analysis=False)
    transformed = ts.get_transformed_data()
    # Transformed series should have less NaNs for first rows due to differencing
    for col in ts.columns:
        assert not transformed[col].isna().all()
        # Should store differencing order
        assert col in ts.diff_order

def test_log_transformation(sample_data):
    ts = TransformTS(sample_data, method='log', analysis=False)
    transformed = ts.get_transformed_data()
    for col in ts.columns:
        # All values should be finite after log transform
        assert np.isfinite(transformed[col]).all()
        # Check is_log flag
        assert ts.is_log[col] is True or ts.is_log[col] is False

def test_log_diff_transformation(sample_data):
    ts = TransformTS(sample_data, method='log-diff', analysis=False)
    transformed = ts.get_transformed_data()
    for col in ts.columns:
        # Differenced values: first value may be NaN, rest finite
        assert np.isfinite(transformed[col]).all()
        assert ts.diff_order[col] == 1

def test_boxcox_transformation(sample_data):
    ts = TransformTS(sample_data, method='boxcox', analysis=False)
    transformed = ts.get_transformed_data()
    for col in ts.columns:
        # Box-Cox transformed series should be finite
        assert np.isfinite(transformed[col]).all()
        # Lambda should be stored
        assert col in ts.lambda_boxcox

def test_hp_transformation(sample_data):
    ts = TransformTS(sample_data, method='hp', analysis=False)
    transformed = ts.get_transformed_data()
    for col in ts.columns:
        # HP filter results should be finite
        assert np.isfinite(transformed[col]).all()

def test_inverse_diff(sample_data):
    ts = TransformTS(sample_data, method='diff', analysis=False)
    transformed = ts.get_transformed_data()
    col = ts.columns[0]
    inv = ts.inverse_transform(col)
    # Length should match original
    assert len(inv) == len(sample_data)
    # Values should roughly match original after inverse
    np.testing.assert_allclose(inv.dropna(), sample_data[col].iloc[ts.diff_order[col]:], rtol=1e-1)

def test_inverse_boxcox(sample_data):
    ts = TransformTS(sample_data, method='boxcox', analysis=False)
    transformed = ts.get_transformed_data()
    col = ts.columns[0]
    inv = ts.inverse_transform(col)
    # Values should roughly match original
    np.testing.assert_allclose(inv.dropna(), sample_data[col], rtol=1e-1)

def test_inverse_log(sample_data):
    ts = TransformTS(sample_data, method='log', analysis=False)
    transformed = ts.get_transformed_data()
    col = ts.columns[0]
    inv = ts.inverse_transform(col)
    np.testing.assert_allclose(inv.dropna(), sample_data[col], rtol=1e-1)

def test_inverse_log_diff(sample_data):
    ts = TransformTS(sample_data, method='log-diff', analysis=False)
    transformed = ts.get_transformed_data()
    col = ts.columns[0]
    inv = ts.inverse_transform(col)
    # Length should roughly match original
    assert len(inv) <= len(sample_data)
    # Values should be finite
    assert np.isfinite(inv.dropna()).all()

def test_trns_info_returns_dict(sample_data):
    ts = TransformTS(sample_data, method='diff', analysis=False)
    info = ts.trns_info()
    assert isinstance(info, dict)
    for col in ts.columns:
        assert 'transformation_method' in info[col]
        assert 'differencing_order' in info[col]
        assert 'is_stationary' in info[col]

def test_non_numeric_column_error():
    df = pd.DataFrame({'A': ['x', 'y', 'z']})
    with pytest.raises(ValueError):
        TransformTS(df)

def test_invalid_method_error(sample_data):
    with pytest.raises(ValueError):
        TransformTS(sample_data, method='invalid')

def test_max_diff_validation(sample_data):
    with pytest.raises(ValueError):
        TransformTS(sample_data, max_diff=0)
if __name__ == "__main__":
    pytest.main([__file__])