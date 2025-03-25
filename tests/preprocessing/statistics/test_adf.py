import numpy as np
import pytest
from pypulate.preprocessing.statistics import augmented_dickey_fuller_test

@pytest.fixture
def stationary_series():
    np.random.seed(42)
    n = 1000
    # AR(1) process with coefficient < 1
    series = np.zeros(n)
    series[0] = np.random.normal(0, 1)
    for t in range(1, n):
        series[t] = 0.5 * series[t-1] + np.random.normal(0, 1)
    return series  # Stationary series

@pytest.fixture
def unit_root_series():
    np.random.seed(42)
    n = 1000
    # Random walk (non-stationary)
    return np.cumsum(np.random.normal(0, 1, n))

@pytest.fixture
def trend_stationary_series():
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 10, n)
    trend = 0.1 * t
    noise = np.random.normal(0, 0.1, n)
    return trend + noise  # Trend stationary series

@pytest.fixture
def mean_reverting_series():
    np.random.seed(42)
    n = 1000
    series = np.zeros(n)
    series[0] = 100
    for t in range(1, n):
        series[t] = series[t-1] + 0.3 * (100 - series[t-1]) + np.random.normal(0, 1)
    return series  # Mean-reverting series

def test_stationary_series(stationary_series):
    stat, p_value = augmented_dickey_fuller_test(stationary_series, regression='c')
    assert p_value < 0.05  # Should reject null of unit root

def test_unit_root_series(unit_root_series):
    stat, p_value = augmented_dickey_fuller_test(unit_root_series, regression='c')
    assert p_value > 0.05  # Should not reject null of unit root

def test_trend_stationary_series(trend_stationary_series):
    stat, p_value = augmented_dickey_fuller_test(trend_stationary_series, regression='ct')
    assert p_value < 0.05  # Should reject null of unit root when including trend

def test_mean_reverting_series(mean_reverting_series):
    stat, p_value = augmented_dickey_fuller_test(mean_reverting_series, regression='c')
    assert p_value < 0.05  # Should reject null of unit root

def test_constant_series():
    constant_data = np.ones(100)
    stat, p_value = augmented_dickey_fuller_test(constant_data)
    assert stat == 0.0 and p_value == 1.0  # Perfect unit root

def test_insufficient_data():
    data = np.array([1, 2])
    stat, p_value = augmented_dickey_fuller_test(data)
    assert np.isnan(stat) and np.isnan(p_value)

def test_nan_handling():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    stat, p_value = augmented_dickey_fuller_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_regression_types():
    np.random.seed(42)
    data = np.random.randn(100).cumsum()
    # Test all regression types
    for reg in ['n', 'c', 'ct']:
        stat, p_value = augmented_dickey_fuller_test(data, regression=reg)
        assert isinstance(stat, float)
        assert isinstance(p_value, float)

def test_invalid_regression():
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        augmented_dickey_fuller_test(data, regression='invalid')

def test_custom_max_lag():
    data = np.random.randn(100).cumsum()
    stat1, p_value1 = augmented_dickey_fuller_test(data, max_lag=1)
    stat2, p_value2 = augmented_dickey_fuller_test(data, max_lag=5)
    assert (stat1, p_value1) != (stat2, p_value2)  # Different lags should give different results

def test_small_sample_warning():
    data = np.random.randn(20)
    with pytest.warns(RuntimeWarning):
        augmented_dickey_fuller_test(data)

def test_negative_lag():
    data = np.random.randn(100)
    with pytest.warns(RuntimeWarning, match="Error computing ADF test: max_lag must be non-negative"):
        stat, p_value = augmented_dickey_fuller_test(data, max_lag=-1)
        assert isinstance(stat, float)
        assert isinstance(p_value, float) 