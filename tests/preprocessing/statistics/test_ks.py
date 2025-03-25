import pytest
import numpy as np
import warnings
from pypulate.preprocessing.statistics import kolmogorov_smirnov_test

# Test fixtures
@pytest.fixture
def normal_data():
    """Generate data from a normal distribution"""
    np.random.seed(42)
    return np.random.normal(loc=2.0, scale=1.5, size=1000)

@pytest.fixture
def uniform_data():
    """Generate data from a uniform distribution"""
    np.random.seed(42)
    return np.random.uniform(low=0.0, high=5.0, size=1000)

@pytest.fixture
def exponential_data():
    """Generate data from an exponential distribution"""
    np.random.seed(42)
    return np.random.exponential(scale=2.0, size=1000)

# Basic functionality tests
def test_normal_distribution(normal_data):
    """Test that normal data is correctly identified"""
    stat, p_value = kolmogorov_smirnov_test(normal_data, dist='norm')
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert 0 <= stat <= 1
    assert 0 <= p_value <= 1
    assert p_value > 0.05  # Should not reject normality

def test_uniform_distribution(uniform_data):
    """Test that uniform data is correctly identified"""
    stat, p_value = kolmogorov_smirnov_test(uniform_data, dist='uniform')
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert 0 <= stat <= 1
    assert 0 <= p_value <= 1
    assert p_value > 0.05  # Should not reject uniformity

def test_exponential_distribution(exponential_data):
    """Test that exponential data is correctly identified"""
    stat, p_value = kolmogorov_smirnov_test(exponential_data, dist='expon')
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert 0 <= stat <= 1
    assert 0 <= p_value <= 1
    assert p_value > 0.05  # Should not reject exponential

def test_wrong_distribution(normal_data):
    """Test that wrong distribution is correctly rejected"""
    stat, p_value = kolmogorov_smirnov_test(normal_data, dist='uniform')
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert 0 <= stat <= 1
    assert 0 <= p_value <= 1
    assert p_value < 0.05  # Should reject wrong distribution

# Parameter tests
def test_custom_params():
    """Test with custom distribution parameters"""
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=2.0, size=1000)
    params = {'loc': 5.0, 'scale': 2.0}
    stat, p_value = kolmogorov_smirnov_test(data, dist='norm', params=params)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert 0 <= stat <= 1
    assert 0 <= p_value <= 1
    assert p_value > 0.05  # Should not reject with correct parameters

def test_invalid_distribution():
    """Test behavior with invalid distribution name"""
    data = np.random.normal(0, 1, 100)
    with pytest.raises(ValueError, match="dist must be one of: 'norm', 'uniform', 'expon'"):
        kolmogorov_smirnov_test(data, dist='invalid')

# Edge cases and error handling
def test_constant_data():
    """Test behavior with constant data"""
    data = np.ones(100)
    stat, p_value = kolmogorov_smirnov_test(data, dist='norm')
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert stat > 0
    assert p_value < 0.05  # Should reject normality for constant data

def test_insufficient_data():
    """Test behavior with insufficient data"""
    data = np.random.normal(0, 1, 2)
    stat, p_value = kolmogorov_smirnov_test(data)
    assert np.isnan(stat)
    assert np.isnan(p_value)

# NaN handling tests
def test_all_nan():
    """Test behavior with all NaN values"""
    data = np.full(100, np.nan)
    stat, p_value = kolmogorov_smirnov_test(data)
    assert np.isnan(stat)
    assert np.isnan(p_value)

def test_some_nan():
    """Test behavior with some NaN values"""
    data = np.random.normal(0, 1, 35)
    data[::2] = np.nan  # Set every other value to NaN
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stat, p_value = kolmogorov_smirnov_test(data)
        assert len(w) >= 1
        assert any("less than 30" in str(warn.message) for warn in w)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_leading_trailing_nan():
    """Test behavior with NaN values at start and end"""
    data = np.random.normal(0, 1, 39)  # 39 total values
    data[:5] = np.nan  # Leading NaNs
    data[-5:] = np.nan  # Trailing NaNs
    # After removing 10 NaN values, we have 29 samples, which should trigger the warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stat, p_value = kolmogorov_smirnov_test(data)
        assert len(w) >= 1
        assert any("less than 30" in str(warn.message) for warn in w)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert np.isfinite(stat)
    assert np.isfinite(p_value)
    assert 0 <= p_value <= 1

# Input type tests
def test_list_input():
    """Test that function accepts list input"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    stat, p_value = kolmogorov_smirnov_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_tuple_input():
    """Test that function accepts tuple input"""
    data = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    stat, p_value = kolmogorov_smirnov_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_small_sample_warning():
    """Test warning for small sample size"""
    data = np.random.normal(0, 1, 20)
    with warnings.catch_warnings(record=True) as w:
        kolmogorov_smirnov_test(data)
        assert len(w) >= 1
        assert any("less than 30" in str(warn.message) for warn in w)

# Performance test
def test_large_input():
    """Test performance with large input"""
    data = np.random.normal(0, 1, 10000)
    stat, p_value = kolmogorov_smirnov_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)
    assert np.isfinite(stat)
    assert np.isfinite(p_value)

# Distribution-specific tests
def test_normal_params_estimation():
    """Test automatic parameter estimation for normal distribution"""
    np.random.seed(42)
    data = np.random.normal(loc=3.0, scale=2.0, size=1000)
    stat, p_value = kolmogorov_smirnov_test(data, dist='norm')
    assert p_value > 0.05  # Should not reject normality with estimated parameters

def test_uniform_params_estimation():
    """Test automatic parameter estimation for uniform distribution"""
    np.random.seed(42)
    data = np.random.uniform(low=1.0, high=4.0, size=1000)
    stat, p_value = kolmogorov_smirnov_test(data, dist='uniform')
    assert p_value > 0.05  # Should not reject uniformity with estimated parameters

def test_exponential_params_estimation():
    """Test automatic parameter estimation for exponential distribution"""
    np.random.seed(42)
    data = np.random.exponential(scale=2.0, size=1000)
    stat, p_value = kolmogorov_smirnov_test(data, dist='expon')
    assert p_value > 0.05  # Should not reject exponential with estimated parameters 