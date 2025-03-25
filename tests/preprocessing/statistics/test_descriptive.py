import numpy as np
import pytest
from pypulate.preprocessing.statistics import descriptive_stats


def test_basic_stats():
    """Test basic descriptive statistics calculation."""
    data = np.array([1, 2, 3, 4, 5])
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['mean'], 3.0)
    assert np.isclose(stats['median'], 3.0)
    assert np.isclose(stats['std'], np.std(data, ddof=1))
    assert np.isclose(stats['min'], 1.0)
    assert np.isclose(stats['max'], 5.0)
    assert np.isclose(stats['count'], 5)
    assert np.isclose(stats['skewness'], 0.0, atol=1e-10)  # Symmetric data
    assert np.isclose(stats['kurtosis'], -1.3, atol=0.1)   # Uniform-like distribution


def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([1, np.nan, 3, np.nan, 5])
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['mean'], 3.0)
    assert np.isclose(stats['median'], 3.0)
    assert np.isclose(stats['count'], 3)  # Only non-NaN values
    assert np.isclose(stats['min'], 1.0)
    assert np.isclose(stats['max'], 5.0)
    assert np.isclose(stats['q1'], 2.0)
    assert np.isclose(stats['q3'], 4.0)


def test_constant_data():
    """Test statistics for constant data."""
    data = np.array([2.5, 2.5, 2.5, 2.5])
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['mean'], 2.5)
    assert np.isclose(stats['median'], 2.5)
    assert np.isclose(stats['std'], 0.0)
    assert np.isclose(stats['min'], 2.5)
    assert np.isclose(stats['max'], 2.5)
    assert np.isclose(stats['q1'], 2.5)
    assert np.isclose(stats['q3'], 2.5)
    assert np.isnan(stats['skewness'])  # Undefined for constant data
    assert np.isnan(stats['kurtosis'])  # Undefined for constant data


def test_single_value():
    """Test statistics for single value."""
    data = np.array([3.0])
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['mean'], 3.0)
    assert np.isclose(stats['median'], 3.0)
    assert np.isnan(stats['std'])  # Undefined for single value
    assert np.isclose(stats['min'], 3.0)
    assert np.isclose(stats['max'], 3.0)
    assert np.isclose(stats['count'], 1)
    assert np.isnan(stats['skewness'])  # Undefined for single value
    assert np.isnan(stats['kurtosis'])  # Undefined for single value


def test_all_nan():
    """Test statistics for all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    stats = descriptive_stats(data)
    
    assert np.isnan(stats['mean'])
    assert np.isnan(stats['median'])
    assert np.isnan(stats['std'])
    assert np.isnan(stats['min'])
    assert np.isnan(stats['max'])
    assert np.isclose(stats['count'], 0)
    assert np.isnan(stats['q1'])
    assert np.isnan(stats['q3'])
    assert np.isnan(stats['skewness'])
    assert np.isnan(stats['kurtosis'])


def test_skewed_data():
    """Test statistics for right-skewed data."""
    data = np.array([1, 1, 2, 2, 2, 3, 10])  # Right-skewed
    stats = descriptive_stats(data)
    
    assert stats['skewness'] > 0  # Right-skewed
    assert stats['mean'] > stats['median']  # Right-skewed property
    assert stats['kurtosis'] > 0  # Leptokurtic due to outlier


def test_negative_values():
    """Test statistics with negative values."""
    data = np.array([-5, -3, -1, 0, 2, 4])
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['mean'], -0.5)
    assert np.isclose(stats['median'], -0.5)
    assert np.isclose(stats['min'], -5.0)
    assert np.isclose(stats['max'], 4.0)


def test_large_values():
    """Test statistics with large values."""
    data = np.array([1e6, 2e6, 3e6, 4e6])
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['mean'], 2.5e6)
    assert np.isclose(stats['median'], 2.5e6)
    assert not np.any(np.isinf(list(stats.values())))  # No infinite values


def test_small_values():
    """Test statistics with small values."""
    data = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['mean'], 2.5e-6)
    assert np.isclose(stats['median'], 2.5e-6)
    assert not np.any(np.isclose(list(stats.values()), 0, atol=1e-20))  # No underflow


def test_mixed_scales():
    """Test statistics with mixed scales."""
    data = np.array([1e-6, 1, 1e6])
    stats = descriptive_stats(data)
    
    assert stats['mean'] > stats['median']  # Right-skewed due to large value
    assert not np.any(np.isinf(list(stats.values())))  # No infinite values
    assert stats['skewness'] > 0  # Right-skewed


def test_invalid_input():
    """Test handling of invalid input types."""
    with pytest.raises(ValueError):
        descriptive_stats("not an array")
    
    with pytest.raises(ValueError):
        descriptive_stats(np.array([[1, 2], [3, 4]]))  # 2D array


def test_quartiles():
    """Test quartile calculations."""
    data = np.array(range(101))  # 0 to 100
    stats = descriptive_stats(data)
    
    assert np.isclose(stats['q1'], 25.0)
    assert np.isclose(stats['median'], 50.0)
    assert np.isclose(stats['q3'], 75.0) 