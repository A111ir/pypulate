import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import resample

def test_basic_mean_resample():
    """Test basic mean resampling."""
    data = np.array([1, 2, 3, 4, 5, 6])
    result = resample(data, factor=2, method='mean')
    expected = np.array([1.5, 3.5, 5.5])
    np.testing.assert_array_equal(result, expected)

def test_basic_median_resample():
    """Test basic median resampling."""
    data = np.array([1, 3, 2, 4, 5, 7])
    result = resample(data, factor=2, method='median')
    expected = np.array([2, 3, 6])
    np.testing.assert_array_equal(result, expected)

def test_basic_sum_resample():
    """Test basic sum resampling."""
    data = np.array([1, 2, 3, 4, 5, 6])
    result = resample(data, factor=2, method='sum')
    expected = np.array([3, 7, 11])
    np.testing.assert_array_equal(result, expected)

def test_basic_min_resample():
    """Test basic min resampling."""
    data = np.array([1, 3, 2, 4, 5, 7])
    result = resample(data, factor=2, method='min')
    expected = np.array([1, 2, 5])
    np.testing.assert_array_equal(result, expected)

def test_basic_max_resample():
    """Test basic max resampling."""
    data = np.array([1, 3, 2, 4, 5, 7])
    result = resample(data, factor=2, method='max')
    expected = np.array([3, 4, 7])
    np.testing.assert_array_equal(result, expected)

def test_factor_one():
    """Test that factor=1 returns original data."""
    data = np.array([1, 2, 3, 4])
    result = resample(data, factor=1)
    np.testing.assert_array_equal(result, data)

def test_factor_equals_length():
    """Test when factor equals data length."""
    data = np.array([1, 2, 3, 4])
    result = resample(data, factor=4, method='mean')
    expected = np.array([2.5])
    np.testing.assert_array_equal(result, expected)

def test_factor_larger_than_length():
    """Test when factor is larger than data length."""
    data = np.array([1, 2, 3])
    result = resample(data, factor=4, method='mean')
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)

def test_nan_handling_mean():
    """Test handling of NaN values with mean method."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = resample(data, factor=2, method='mean')
    expected = np.array([1, 3.5, 6])
    np.testing.assert_array_equal(result, expected)

def test_nan_handling_sum():
    """Test handling of NaN values with sum method."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = resample(data, factor=2, method='sum')
    expected = np.array([1, 7, 6])
    np.testing.assert_array_equal(result, expected)

def test_all_nan_group():
    """Test handling of groups with all NaN values."""
    data = np.array([1, 2, np.nan, np.nan, 5, 6])
    result = resample(data, factor=2, method='mean')
    expected = np.array([1.5, np.nan, 5.5])
    np.testing.assert_array_equal(result, expected)

def test_empty_array():
    """Test with empty input array."""
    data = np.array([])
    result = resample(data, factor=2)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)

def test_single_value():
    """Test with single value input."""
    data = np.array([1])
    result = resample(data, factor=2)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)

def test_non_divisible_length():
    """Test with data length not divisible by factor."""
    data = np.array([1, 2, 3, 4, 5])  # Length 5
    result = resample(data, factor=2)  # Should only use first 4 values
    expected = np.array([1.5, 3.5])
    np.testing.assert_array_equal(result, expected)

def test_invalid_method():
    """Test that invalid method raises ValueError."""
    data = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="Method must be one of: 'mean', 'median', 'sum', 'min', 'max'"):
        resample(data, factor=2, method='invalid')

def test_different_input_types():
    """Test with different input types."""
    data = [1, 2, 3, 4]  # list input
    result = resample(data, factor=2)
    expected = np.array([1.5, 3.5])
    np.testing.assert_array_equal(result, expected)

def test_floating_point_data():
    """Test with floating-point data."""
    data = np.array([1.5, 2.5, 3.5, 4.5])
    result = resample(data, factor=2)
    expected = np.array([2.0, 4.0])
    np.testing.assert_array_equal(result, expected) 