import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import min_max_scale

def test_basic_scaling():
    """Test basic min-max scaling."""
    data = np.array([1, 2, 3, 4, 5])
    result = min_max_scale(data)
    np.testing.assert_array_almost_equal(result, [0, 0.25, 0.5, 0.75, 1])

def test_custom_range():
    """Test scaling to custom range."""
    data = np.array([1, 2, 3, 4, 5])
    result = min_max_scale(data, feature_range=(-1, 1))
    np.testing.assert_array_almost_equal(result, [-1, -0.5, 0, 0.5, 1])

def test_with_nan():
    """Test scaling with NaN values."""
    data = np.array([1, 2, np.nan, 4, 5])
    result = min_max_scale(data)
    assert np.isnan(result[2])
    np.testing.assert_array_almost_equal(result[~np.isnan(result)], [0, 0.25, 0.75, 1])

def test_constant_input():
    """Test scaling with constant input."""
    data = np.array([5, 5, 5, 5, 5])
    result = min_max_scale(data)
    np.testing.assert_array_equal(result, np.zeros_like(data))

def test_empty_array():
    """Test scaling with empty array."""
    data = np.array([])
    result = min_max_scale(data)
    assert len(result) == 0

def test_single_value():
    """Test scaling with single value."""
    data = np.array([42])
    result = min_max_scale(data)
    np.testing.assert_array_equal(result, [0])

def test_negative_values():
    """Test scaling with negative values."""
    data = np.array([-5, -4, -3, -2, -1])
    result = min_max_scale(data)
    np.testing.assert_array_almost_equal(result, [0, 0.25, 0.5, 0.75, 1])

def test_mixed_values():
    """Test scaling with mixed positive and negative values."""
    data = np.array([-2, -1, 0, 1, 2])
    result = min_max_scale(data)
    np.testing.assert_array_almost_equal(result, [0, 0.25, 0.5, 0.75, 1])

def test_different_input_types():
    """Test scaling with different input types."""
    data_list = [1, 2, 3, 4, 5]
    result = min_max_scale(data_list)
    expected = min_max_scale(np.array(data_list))
    np.testing.assert_array_equal(result, expected)

def test_floating_point_data():
    """Test scaling with floating-point data."""
    data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    result = min_max_scale(data)
    assert result.dtype == np.float64
    np.testing.assert_array_almost_equal(result, [0, 0.25, 0.5, 0.75, 1])

def test_all_nan():
    """Test scaling with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    result = min_max_scale(data)
    assert np.all(np.isnan(result))

def test_mostly_nan():
    """Test scaling with mostly NaN values."""
    data = np.array([np.nan, np.nan, 1, 2, np.nan])
    result = min_max_scale(data)
    assert np.all(np.isnan(result) == np.isnan(data))
    valid_mask = ~np.isnan(result)
    np.testing.assert_array_almost_equal(result[valid_mask], [0, 1])

def test_invalid_feature_range():
    """Test that invalid feature range raises ValueError."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        min_max_scale(data, feature_range=(1, 0))  # max < min

def test_extreme_values():
    """Test scaling with extreme values."""
    data = np.array([1e-10, 1e10])
    result = min_max_scale(data)
    np.testing.assert_array_almost_equal(result, [0, 1]) 