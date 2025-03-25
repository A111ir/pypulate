import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import fill_missing

def test_basic_mean():
    """Test basic mean filling."""
    data = np.array([1, np.nan, 3, np.nan, 5])
    result = fill_missing(data, method='mean')
    expected = np.array([1, 3, 3, 3, 5])  # mean = 3
    np.testing.assert_array_equal(result, expected)

def test_basic_median():
    """Test basic median filling."""
    data = np.array([1, np.nan, 5, np.nan, 3])
    result = fill_missing(data, method='median')
    expected = np.array([1, 3, 5, 3, 3])  # median = 3
    np.testing.assert_array_equal(result, expected)

def test_basic_mode():
    """Test basic mode filling."""
    data = np.array([1, np.nan, 2, 2, np.nan, 3])
    result = fill_missing(data, method='mode')
    expected = np.array([1, 2, 2, 2, 2, 3])  # mode = 2
    np.testing.assert_array_equal(result, expected)

def test_basic_forward():
    """Test basic forward filling."""
    data = np.array([1, np.nan, np.nan, 4, np.nan])
    result = fill_missing(data, method='forward')
    expected = np.array([1, 1, 1, 4, 4])
    np.testing.assert_array_equal(result, expected)

def test_basic_backward():
    """Test basic backward filling."""
    data = np.array([np.nan, 2, np.nan, np.nan, 5])
    result = fill_missing(data, method='backward')
    expected = np.array([2, 2, 5, 5, 5])
    np.testing.assert_array_equal(result, expected)

def test_basic_value():
    """Test basic value filling."""
    data = np.array([1, np.nan, 3, np.nan])
    result = fill_missing(data, method='value', value=0)
    expected = np.array([1, 0, 3, 0])
    np.testing.assert_array_equal(result, expected)

def test_no_nan():
    """Test with no NaN values."""
    data = np.array([1, 2, 3, 4])
    result = fill_missing(data)
    np.testing.assert_array_equal(result, data)

def test_all_nan():
    """Test with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    # Mean/median/mode should fill with NaN for all NaN input
    result_mean = fill_missing(data, method='mean')
    np.testing.assert_array_equal(result_mean, data)
    
    result_median = fill_missing(data, method='median')
    np.testing.assert_array_equal(result_median, data)
    
    result_mode = fill_missing(data, method='mode')
    np.testing.assert_array_equal(result_mode, data)
    
    # Forward/backward fill should keep NaN
    result_forward = fill_missing(data, method='forward')
    np.testing.assert_array_equal(result_forward, data)
    
    result_backward = fill_missing(data, method='backward')
    np.testing.assert_array_equal(result_backward, data)
    
    # Value fill should work
    result_value = fill_missing(data, method='value', value=0)
    np.testing.assert_array_equal(result_value, np.zeros_like(data))

def test_single_value():
    """Test with single value."""
    data = np.array([np.nan])
    result = fill_missing(data, method='value', value=1)
    expected = np.array([1])
    np.testing.assert_array_equal(result, expected)

def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    result = fill_missing(data)
    np.testing.assert_array_equal(result, data)

def test_different_input_types():
    """Test with different input types."""
    data = [1, None, 3]  # list input with None
    result = fill_missing(data, method='mean')
    expected = np.array([1, 2, 3])  # mean = 2
    np.testing.assert_array_equal(result, expected)

def test_floating_point_data():
    """Test with floating-point data."""
    data = np.array([1.5, np.nan, 3.5])
    result = fill_missing(data, method='mean')
    expected = np.array([1.5, 2.5, 3.5])  # mean = 2.5
    np.testing.assert_array_equal(result, expected)

def test_invalid_method():
    """Test that invalid method raises ValueError."""
    data = np.array([1, np.nan, 3])
    with pytest.raises(ValueError, match="Method must be one of:"):
        fill_missing(data, method='invalid')

def test_value_without_value():
    """Test value method without providing value."""
    data = np.array([1, np.nan, 3])
    with pytest.raises(ValueError, match="Value must be provided"):
        fill_missing(data, method='value')

def test_forward_fill_start():
    """Test forward filling at start of array."""
    data = np.array([np.nan, np.nan, 3, 4])
    result = fill_missing(data, method='forward')
    expected = np.array([np.nan, np.nan, 3, 4])  # Can't forward fill start
    np.testing.assert_array_equal(result, expected)

def test_backward_fill_end():
    """Test backward filling at end of array."""
    data = np.array([1, 2, np.nan, np.nan])
    result = fill_missing(data, method='backward')
    expected = np.array([1, 2, np.nan, np.nan])  # Can't backward fill end
    np.testing.assert_array_equal(result, expected)

def test_mode_multiple_modes():
    """Test mode filling with multiple modes."""
    data = np.array([1, 1, np.nan, 2, 2, np.nan])
    result = fill_missing(data, method='mode')
    # Should use first mode encountered
    assert result[2] in [1, 2]
    assert result[5] in [1, 2] 