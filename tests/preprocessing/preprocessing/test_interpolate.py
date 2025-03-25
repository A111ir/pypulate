import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import interpolate_missing

def test_basic_linear():
    """Test basic linear interpolation."""
    data = np.array([1, np.nan, 3])
    result = interpolate_missing(data, method='linear')
    expected = np.array([1, 2, 3])
    np.testing.assert_array_almost_equal(result, expected)

def test_basic_nearest():
    """Test basic nearest interpolation."""
    data = np.array([1, np.nan, np.nan, 4])
    result = interpolate_missing(data, method='nearest')
    expected = np.array([1, 1, 4, 4])
    np.testing.assert_array_equal(result, expected)

def test_basic_zero():
    """Test basic zero interpolation."""
    data = np.array([1, np.nan, np.nan, 4])
    result = interpolate_missing(data, method='zero')
    expected = np.array([1, 1, 1, 4])
    np.testing.assert_array_equal(result, expected)

def test_basic_slinear():
    """Test basic spline linear interpolation."""
    data = np.array([1, np.nan, 3, np.nan, 5])
    result = interpolate_missing(data, method='slinear')
    expected = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_almost_equal(result, expected)

def test_basic_quadratic():
    """Test basic quadratic interpolation."""
    data = np.array([1, np.nan, 4, np.nan, 9])
    result = interpolate_missing(data, method='quadratic')
    expected = np.array([1, 2.25, 4, 6.25, 9])
    np.testing.assert_array_almost_equal(result, expected)

def test_basic_cubic():
    """Test basic cubic interpolation."""
    data = np.array([1, np.nan, 8, 16, np.nan, 27, 32])
    result = interpolate_missing(data, method='cubic')
    # We don't test exact values because cubic interpolation can vary
    # Instead, we test that:
    # 1. The result has the same shape
    # 2. NaN values are filled
    # 3. Original values are preserved
    assert len(result) == len(data)
    assert not np.any(np.isnan(result))
    np.testing.assert_array_almost_equal(result[np.logical_not(np.isnan(data))],
                                       data[np.logical_not(np.isnan(data))])

def test_no_nan():
    """Test with no NaN values."""
    data = np.array([1, 2, 3, 4])
    result = interpolate_missing(data)
    np.testing.assert_array_equal(result, data)

def test_all_nan():
    """Test with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    result = interpolate_missing(data)
    np.testing.assert_array_equal(result, data)

def test_single_value():
    """Test with single value."""
    data = np.array([1])
    result = interpolate_missing(data)
    np.testing.assert_array_equal(result, data)

def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    result = interpolate_missing(data)
    np.testing.assert_array_equal(result, data)

def test_start_nan():
    """Test with NaN values at start."""
    data = np.array([np.nan, np.nan, 3, 4])
    # Linear and other methods should keep NaN at start
    result_linear = interpolate_missing(data, method='linear')
    expected_linear = np.array([np.nan, np.nan, 3, 4])
    np.testing.assert_array_equal(result_linear, expected_linear)
    
    # Nearest should fill with nearest valid value
    result_nearest = interpolate_missing(data, method='nearest')
    expected_nearest = np.array([3, 3, 3, 4])
    np.testing.assert_array_equal(result_nearest, expected_nearest)

def test_end_nan():
    """Test with NaN values at end."""
    data = np.array([1, 2, np.nan, np.nan])
    # Linear and other methods should keep NaN at end
    result_linear = interpolate_missing(data, method='linear')
    expected_linear = np.array([1, 2, np.nan, np.nan])
    np.testing.assert_array_equal(result_linear, expected_linear)
    
    # Nearest should fill with nearest valid value
    result_nearest = interpolate_missing(data, method='nearest')
    expected_nearest = np.array([1, 2, 2, 2])
    np.testing.assert_array_equal(result_nearest, expected_nearest)

def test_different_input_types():
    """Test with different input types."""
    data = [1, None, 3]  # list input with None
    result = interpolate_missing(data)
    expected = np.array([1, 2, 3])
    np.testing.assert_array_almost_equal(result, expected)

def test_floating_point_data():
    """Test with floating-point data."""
    data = np.array([1.5, np.nan, 3.5])
    result = interpolate_missing(data)
    expected = np.array([1.5, 2.5, 3.5])
    np.testing.assert_array_almost_equal(result, expected)

def test_invalid_method():
    """Test that invalid method raises ValueError."""
    data = np.array([1, np.nan, 3])
    with pytest.raises(ValueError, match="Method must be one of:"):
        interpolate_missing(data, method='invalid')

def test_sparse_data():
    """Test with sparse valid data points."""
    data = np.array([1, np.nan, np.nan, np.nan, 5])
    # Linear interpolation
    result_linear = interpolate_missing(data, method='linear')
    expected_linear = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_almost_equal(result_linear, expected_linear)
    
    # Nearest interpolation
    result_nearest = interpolate_missing(data, method='nearest')
    expected_nearest = np.array([1, 1, 1, 5, 5])
    np.testing.assert_array_equal(result_nearest, expected_nearest)

def test_alternating_nan():
    """Test with alternating NaN values."""
    data = np.array([1, np.nan, 3, np.nan, 5, np.nan])
    result = interpolate_missing(data)
    expected = np.array([1, 2, 3, 4, 5, np.nan])  # Last NaN should remain
    np.testing.assert_array_almost_equal(result, expected) 