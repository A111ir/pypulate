import pytest
import numpy as np
import warnings
from pypulate.preprocessing.statistics import hurst_exponent

# Test fixtures
@pytest.fixture
def random_walk():
    """Generate a random walk (H ≈ 0.5)"""
    np.random.seed(42)
    n = 1000
    # Generate random steps
    steps = np.random.choice([-1, 1], size=n)
    # Generate random walk through cumulative sum
    return np.cumsum(steps)

@pytest.fixture
def trending_series():
    """Generate a trending series (H > 0.5)"""
    np.random.seed(42)
    n = 1000
    # Generate strong trend with small noise
    t = np.linspace(0, 10, n)
    trend = t**2  # Quadratic trend
    noise = np.random.normal(0, 0.1, n)
    return trend + noise

@pytest.fixture
def mean_reverting_series():
    """Generate a mean-reverting series (H < 0.5)"""
    np.random.seed(42)
    n = 1000
    series = np.zeros(n)
    series[0] = np.random.normal(0, 1)
    for i in range(1, n):
        series[i] = -0.7 * series[i-1] + np.random.normal(0, 1)
    return series

# Basic functionality tests
def test_random_walk(random_walk):
    """Test that random walk has H ≈ 0.5"""
    h = hurst_exponent(random_walk)
    assert isinstance(h, float)
    assert 0.4 <= h <= 0.6  # Allow some estimation error

def test_trending_series(trending_series):
    """Test that trending series has H > 0.5"""
    h = hurst_exponent(trending_series)
    assert isinstance(h, float)
    assert h > 0.5

def test_mean_reverting_series(mean_reverting_series):
    """Test that mean-reverting series has H < 0.5"""
    h = hurst_exponent(mean_reverting_series)
    assert isinstance(h, float)
    assert h < 0.5

def test_custom_max_lag():
    """Test behavior with custom max_lag"""
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    h = hurst_exponent(data, max_lag=10)
    assert isinstance(h, float)
    assert 0 <= h <= 1

# Edge cases and error handling
def test_constant_series():
    """Test behavior with constant series"""
    data = np.ones(100)
    h = hurst_exponent(data)
    assert np.isnan(h)

def test_insufficient_data():
    """Test behavior with insufficient data"""
    data = np.random.normal(0, 1, 9)  # Less than minimum required (10)
    h = hurst_exponent(data)
    assert np.isnan(h)

def test_small_sample_warning():
    """Test warning for small sample size"""
    data = np.random.normal(0, 1, 50)
    with warnings.catch_warnings(record=True) as w:
        hurst_exponent(data)
        assert len(w) >= 1
        assert any("less than 100" in str(warn.message) for warn in w)

# NaN handling tests
def test_all_nan():
    """Test behavior with all NaN values"""
    data = np.full(100, np.nan)
    h = hurst_exponent(data)
    assert np.isnan(h)

def test_some_nan():
    """Test behavior with some NaN values"""
    data = np.random.normal(0, 1, 150)
    data[::3] = np.nan  # Set every third value to NaN
    with warnings.catch_warnings(record=True) as w:
        h = hurst_exponent(data)
        assert isinstance(h, float)
        assert 0 <= h <= 1

def test_leading_trailing_nan():
    """Test behavior with NaN values at start and end"""
    data = np.random.normal(0, 1, 150)
    data[:10] = np.nan  # Leading NaNs
    data[-10:] = np.nan  # Trailing NaNs
    with warnings.catch_warnings(record=True) as w:
        h = hurst_exponent(data)
        assert isinstance(h, float)
        assert 0 <= h <= 1

# Input type tests
def test_list_input():
    """Test that function accepts list input"""
    data = [float(x) for x in range(100)]
    h = hurst_exponent(data)
    assert isinstance(h, float)
    assert 0 <= h <= 1

def test_tuple_input():
    """Test that function accepts tuple input"""
    data = tuple(float(x) for x in range(100))
    h = hurst_exponent(data)
    assert isinstance(h, float)
    assert 0 <= h <= 1

# Performance test
def test_large_input():
    """Test performance with large input"""
    data = np.random.normal(0, 1, 10000)
    h = hurst_exponent(data)
    assert isinstance(h, float)
    assert 0 <= h <= 1

# Special cases
def test_alternating_series():
    """Test behavior with alternating series"""
    data = np.array([1, -1] * 100)  # Strong mean reversion
    h = hurst_exponent(data)
    assert isinstance(h, float)
    assert h < 0.5  # Should indicate mean reversion

def test_strong_trend():
    """Test behavior with strong trend"""
    t = np.linspace(0, 10, 200)
    data = t**2  # Quadratic trend
    h = hurst_exponent(data)
    assert isinstance(h, float)
    assert h > 0.5  # Should indicate trend

def test_noisy_periodic():
    """Test behavior with noisy periodic data"""
    t = np.linspace(0, 10, 200)
    data = np.sin(t) + np.random.normal(0, 0.1, 200)
    h = hurst_exponent(data)
    assert isinstance(h, float)
    assert 0 <= h <= 1 