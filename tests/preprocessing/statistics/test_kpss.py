import numpy as np
import pytest
from pypulate.preprocessing.statistics import kpss_test

@pytest.fixture
def stationary_series():
    np.random.seed(42)
    n = 1000
    return np.random.normal(100, 1, n)  # Stationary around constant mean

@pytest.fixture
def trend_stationary_series():
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 10, n)
    trend = 0.1 * t
    noise = np.random.normal(0, 0.1, n)
    return trend + noise  # Stationary around linear trend

@pytest.fixture
def unit_root_series():
    np.random.seed(42)
    n = 1000
    return np.cumsum(np.random.normal(0, 1, n))  # Random walk (non-stationary)

def test_stationary_around_constant(stationary_series):
    stat, p_value = kpss_test(stationary_series, regression='c')
    assert p_value > 0.05  # Should not reject null of level stationarity

def test_trend_stationary(trend_stationary_series):
    stat, p_value = kpss_test(trend_stationary_series, regression='ct')
    assert p_value > 0.05  # Should not reject null of trend stationarity

def test_unit_root(unit_root_series):
    stat, p_value = kpss_test(unit_root_series, regression='c')
    assert p_value < 0.05  # Should reject null of stationarity

def test_constant_series():
    constant_data = np.ones(100)
    stat, p_value = kpss_test(constant_data, regression='c')
    assert p_value == 1.0  # Perfectly stationary

def test_insufficient_data():
    data = np.array([1, 2])
    stat, p_value = kpss_test(data)
    assert np.isnan(stat) and np.isnan(p_value)

def test_nan_handling():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    stat, p_value = kpss_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_regression_type_validation():
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        kpss_test(data, regression='invalid')

def test_custom_lags():
    np.random.seed(42)
    data = np.random.randn(100)
    stat1, p_value1 = kpss_test(data, lags=5)
    stat2, p_value2 = kpss_test(data, lags=10)
    assert (stat1, p_value1) != (stat2, p_value2)  # Different lags should give different results

def test_small_sample_warning():
    data = np.random.randn(20)
    with pytest.warns(RuntimeWarning):
        kpss_test(data)

def test_both_regression_types():
    np.random.seed(42)
    data = np.random.randn(100)
    level_stat, level_p = kpss_test(data, regression='c')
    trend_stat, trend_p = kpss_test(data, regression='ct')
    assert (level_stat, level_p) != (trend_stat, trend_p)  # Different regression types should give different results 