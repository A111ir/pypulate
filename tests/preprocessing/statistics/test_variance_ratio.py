import numpy as np
import pytest
from pypulate.preprocessing.statistics import variance_ratio_test

@pytest.fixture
def random_walk():
    np.random.seed(42)
    n = 1000
    steps = np.random.choice([-1, 1], size=n)
    return np.exp(np.cumsum(steps) * 0.01)  # Small steps for price-like behavior

@pytest.fixture
def trending_series():
    np.random.seed(42)
    n = 1000
    t = np.arange(n)
    trend = np.exp(0.001 * t)  # Exponential trend
    noise = np.random.normal(0, 0.01, n)
    return trend * (1 + noise)

@pytest.fixture
def mean_reverting_series():
    np.random.seed(42)
    n = 1000
    x = np.zeros(n)
    x[0] = 100
    for i in range(1, n):
        x[i] = x[i-1] + 0.5 * (100 - x[i-1]) + np.random.normal(0, 0.5)  # Stronger mean reversion
    return np.exp(x * 0.01)  # Convert to price-like positive values

def test_random_walk(random_walk):
    results = variance_ratio_test(random_walk)
    for period, (stat, p_value) in results.items():
        assert -2 < stat < 2  # Test statistic should be close to 0
        assert p_value > 0.01  # Should not reject random walk hypothesis

def test_trending_series(trending_series):
    results = variance_ratio_test(trending_series)
    significant_trends = sum(p_value < 0.05 for _, p_value in results.values())
    assert significant_trends > 0  # Should reject random walk for some periods

def test_mean_reverting_series(mean_reverting_series):
    results = variance_ratio_test(mean_reverting_series)
    significant_reversions = sum(p_value < 0.05 for _, p_value in results.values())
    assert significant_reversions > 0  # Should reject random walk for some periods

def test_constant_series():
    constant_data = np.ones(100)
    results = variance_ratio_test(constant_data)
    for period, (stat, p_value) in results.items():
        assert stat == 0.0
        assert p_value == 1.0

def test_custom_periods():
    np.random.seed(42)
    data = np.exp(np.random.randn(200).cumsum())
    periods = [2, 5, 10]
    results = variance_ratio_test(data, periods=periods)
    assert set(results.keys()) == set(periods)

def test_insufficient_data():
    with pytest.raises(ValueError):
        variance_ratio_test(np.array([1, 2]))

def test_nan_handling():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    results = variance_ratio_test(data)
    assert all(isinstance(stat, float) for stat, _ in results.values())
    assert all(isinstance(p_value, float) for _, p_value in results.values())

def test_negative_values():
    with pytest.raises(ValueError):
        variance_ratio_test(np.array([-1.0, 2.0, 3.0] * 20))

def test_robust_option():
    np.random.seed(42)
    data = np.exp(np.random.randn(100).cumsum())
    robust_results = variance_ratio_test(data, robust=True)
    non_robust_results = variance_ratio_test(data, robust=False)
    # Results should be different when using robust standard errors
    assert any(robust_results[k] != non_robust_results[k] for k in robust_results)

def test_large_sample():
    np.random.seed(42)
    data = np.exp(np.random.randn(1000).cumsum())
    results = variance_ratio_test(data)
    assert all(isinstance(stat, float) for stat, _ in results.values())
    assert all(isinstance(p_value, float) for _, p_value in results.values()) 