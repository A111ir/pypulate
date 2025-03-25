import numpy as np
import pytest
from pypulate.preprocessing.statistics import ljung_box_test

@pytest.fixture
def independent_series():
    np.random.seed(42)
    return np.random.normal(0, 1, 1000)  # Independent observations

@pytest.fixture
def autocorrelated_series():
    np.random.seed(42)
    n = 1000
    ar_coef = 0.8
    series = np.zeros(n)
    series[0] = np.random.normal(0, 1)
    for t in range(1, n):
        series[t] = ar_coef * series[t-1] + np.random.normal(0, np.sqrt(1 - ar_coef**2))
    return series  # AR(1) process with strong autocorrelation

@pytest.fixture
def seasonal_series():
    np.random.seed(42)
    n = 1000
    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / 12)  # Period of 12
    noise = np.random.normal(0, 0.1, n)
    return seasonal + noise

def test_independent_series(independent_series):
    stat, p_value = ljung_box_test(independent_series)
    assert p_value > 0.05  # Should not reject null of independence

def test_autocorrelated_series(autocorrelated_series):
    stat, p_value = ljung_box_test(autocorrelated_series)
    assert p_value < 0.05  # Should reject null of independence

def test_seasonal_series(seasonal_series):
    stat, p_value = ljung_box_test(seasonal_series, lags=24)  # Test with 2 seasonal periods
    assert p_value < 0.05  # Should reject null of independence

def test_constant_series():
    constant_data = np.ones(100)
    result = ljung_box_test(constant_data)
    assert isinstance(result, dict)
    for lag, (stat, p_value) in result.items():
        assert stat == 0.0 and p_value == 1.0  # Perfect autocorrelation for constant series

def test_insufficient_data():
    data = np.array([1, 2])
    stat, p_value = ljung_box_test(data)
    assert np.isnan(stat) and np.isnan(p_value)

def test_nan_handling():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    stat, p_value = ljung_box_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_different_lags():
    np.random.seed(42)
    data = np.random.randn(100)
    stat1, p_value1 = ljung_box_test(data, lags=5)
    stat2, p_value2 = ljung_box_test(data, lags=10)
    assert (stat1, p_value1) != (stat2, p_value2)  # Different lags should give different results

def test_boxpierce_option():
    np.random.seed(42)
    data = np.random.randn(100)
    lb_stat, lb_p = ljung_box_test(data, boxpierce=False)
    bp_stat, bp_p = ljung_box_test(data, boxpierce=True)
    assert (lb_stat, lb_p) != (bp_stat, bp_p)  # Ljung-Box and Box-Pierce should differ

def test_small_sample_warning():
    data = np.random.randn(20)
    with pytest.warns(RuntimeWarning):
        ljung_box_test(data, lags=10)  # Warning for small sample size relative to lags

def test_invalid_lags():
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        ljung_box_test(data, lags=0)  # Should raise error for non-positive lags 