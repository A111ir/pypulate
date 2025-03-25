import pytest
import numpy as np
import warnings
from pypulate.preprocessing.statistics import arch_test

# Test fixtures
@pytest.fixture
def arch_process():
    """Generate returns from an ARCH(1) process"""
    np.random.seed(42)
    n = 1000
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    omega = 0.1
    alpha = 0.7
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2
        returns[t] = np.sqrt(sigma2[t]) * np.random.normal(0, 1)
    
    return returns

@pytest.fixture
def homoskedastic_process():
    """Generate returns from a homoskedastic process"""
    np.random.seed(42)
    return np.random.normal(0, 1, 1000)

# Basic functionality tests
def test_arch_effect_detection(arch_process):
    """Test that ARCH effects are detected in an ARCH process"""
    stat, p_value = arch_test(arch_process)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert stat > 0
    assert 0 <= p_value <= 1
    assert p_value < 0.05  # Should reject null of no ARCH effects

def test_homoskedastic_process(homoskedastic_process):
    """Test that homoskedastic process shows no ARCH effects"""
    stat, p_value = arch_test(homoskedastic_process)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert stat > 0
    assert 0 <= p_value <= 1
    assert p_value > 0.01  # Should not strongly reject null of no ARCH effects

def test_different_lags():
    """Test behavior with different numbers of lags"""
    np.random.seed(42)
    returns = np.random.normal(0, 1, 100)
    
    # Test with various lag values
    for lags in [1, 5, 10]:
        stat, p_value = arch_test(returns, lags=lags)
        assert isinstance(stat, float)
        assert isinstance(p_value, float)
        assert stat >= 0
        assert 0 <= p_value <= 1

# Edge cases and error handling
def test_constant_returns():
    """Test behavior with constant returns"""
    returns = np.ones(100)
    stat, p_value = arch_test(returns)
    assert np.isnan(stat)
    assert np.isnan(p_value)

def test_zero_returns():
    """Test behavior with zero returns"""
    returns = np.zeros(100)
    stat, p_value = arch_test(returns)
    assert np.isnan(stat)
    assert np.isnan(p_value)

def test_insufficient_data():
    """Test behavior with insufficient data"""
    returns = np.random.normal(0, 1, 4)  # Less than default lags
    stat, p_value = arch_test(returns)
    assert np.isnan(stat)
    assert np.isnan(p_value)

def test_invalid_lags():
    """Test behavior with invalid lags parameter"""
    returns = np.random.normal(0, 1, 100)
    with pytest.raises(ValueError, match="lags must be positive"):
        arch_test(returns, lags=0)
    with pytest.raises(ValueError, match="lags must be positive"):
        arch_test(returns, lags=-1)

# NaN handling tests
def test_all_nan():
    """Test behavior with all NaN values"""
    returns = np.full(100, np.nan)
    stat, p_value = arch_test(returns)
    assert np.isnan(stat)
    assert np.isnan(p_value)

def test_some_nan():
    """Test behavior with some NaN values"""
    returns = np.random.normal(0, 1, 35)
    returns[::2] = np.nan  # Set every other value to NaN
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stat, p_value = arch_test(returns)
        assert len(w) >= 1
        assert any("less than 30" in str(warn.message) for warn in w)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_leading_trailing_nan():
    """Test behavior with NaN values at start and end"""
    returns = np.random.normal(0, 1, 40)
    returns[:5] = np.nan  # Leading NaNs
    returns[-5:] = np.nan  # Trailing NaNs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stat, p_value = arch_test(returns)
        # After removing 10 NaN values, we have 30 samples, which is at the threshold
        # So we should not expect a warning
        assert isinstance(stat, float)
        assert isinstance(p_value, float)
        assert np.isfinite(stat)
        assert np.isfinite(p_value)
        assert 0 <= p_value <= 1

# Input type tests
def test_list_input():
    """Test that function accepts list input"""
    returns = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    stat, p_value = arch_test(returns)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_tuple_input():
    """Test that function accepts tuple input"""
    returns = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    stat, p_value = arch_test(returns)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_small_sample_warning():
    """Test warning for small sample size"""
    returns = np.random.normal(0, 1, 20)
    with warnings.catch_warnings(record=True) as w:
        arch_test(returns)
        assert len(w) >= 1
        assert any("less than 30" in str(warn.message) for warn in w)

# Performance test
def test_large_input():
    """Test performance with large input"""
    returns = np.random.normal(0, 1, 10000)
    stat, p_value = arch_test(returns)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert np.isfinite(stat)
    assert np.isfinite(p_value) 