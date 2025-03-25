import numpy as np
import pytest
from pypulate.preprocessing.statistics import granger_causality_test

@pytest.fixture
def independent_series():
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)
    return x, y  # Two independent series

@pytest.fixture
def causal_series():
    np.random.seed(42)
    n = 1000
    x = np.random.normal(0, 1, n)
    y = np.zeros(n)
    y[0] = np.random.normal(0, 1)
    # y depends on lagged values of x
    for t in range(1, n):
        y[t] = 0.5 * x[t-1] + 0.2 * np.random.normal(0, 1)
    return x, y  # x Granger-causes y

@pytest.fixture
def bidirectional_series():
    np.random.seed(42)
    n = 1000
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = np.random.normal(0, 1)
    y[0] = np.random.normal(0, 1)
    # x and y affect each other
    for t in range(1, n):
        x[t] = 0.3 * y[t-1] + np.random.normal(0, 0.5)
        y[t] = 0.3 * x[t-1] + np.random.normal(0, 0.5)
    return x, y

def test_independent_series(independent_series):
    x, y = independent_series
    stat, p_value = granger_causality_test(x, y)
    assert p_value > 0.05  # Should not reject null of no Granger causality

def test_causal_series(causal_series):
    x, y = causal_series
    stat, p_value = granger_causality_test(x, y)
    assert p_value < 0.05  # Should reject null of no Granger causality

def test_bidirectional_series(bidirectional_series):
    x, y = bidirectional_series
    # Test x -> y
    stat1, p_value1 = granger_causality_test(x, y)
    # Test y -> x
    stat2, p_value2 = granger_causality_test(y, x)
    assert p_value1 < 0.05 and p_value2 < 0.05  # Both directions should show causality

def test_constant_series():
    x = np.ones(100)
    y = np.random.normal(0, 1, 100)
    stat, p_value = granger_causality_test(x, y)
    assert stat == 0.0 and p_value == 1.0  # No causality for constant series

def test_insufficient_data():
    x = np.array([1, 2])
    y = np.array([1, 2])
    stat, p_value = granger_causality_test(x, y)
    assert np.isnan(stat) and np.isnan(p_value)

def test_nan_handling():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    y = np.array([2.0, 3.0, np.nan, 5.0, 6.0] * 20)
    stat, p_value = granger_causality_test(x, y)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_different_lags():
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.roll(x, 1) + 0.1 * np.random.randn(100)
    stat1, p_value1 = granger_causality_test(x, y, max_lag=1)
    stat2, p_value2 = granger_causality_test(x, y, max_lag=2)
    assert (stat1, p_value1) != (stat2, p_value2)  # Different lags should give different results

def test_shape_mismatch():
    x = np.random.randn(100)
    y = np.random.randn(50)
    with pytest.raises(ValueError):
        granger_causality_test(x, y)

def test_small_sample_warning():
    x = np.random.randn(20)
    y = np.random.randn(20)
    with pytest.warns(RuntimeWarning):
        granger_causality_test(x, y, max_lag=5)

def test_invalid_lag():
    x = np.random.randn(100)
    y = np.random.randn(100)
    with pytest.raises(ValueError):
        granger_causality_test(x, y, max_lag=0) 