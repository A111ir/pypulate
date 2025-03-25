import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import robust_scale

def test_basic_iqr():
    """Test basic IQR scaling."""
    data = np.array([1, 2, 3, 4, 5])
    result = robust_scale(data)
    # For this data:
    # median = 3
    # Q1 = 2 (25th percentile)
    # Q3 = 4 (75th percentile)
    # IQR = 2
    expected = (data - 3) / 2
    np.testing.assert_array_almost_equal(result, expected)

def test_basic_mad():
    """Test basic MAD scaling."""
    data = np.array([1, 2, 3, 4, 5])
    result = robust_scale(data, method='mad')
    # For this data:
    # median = 3
    # MAD = median(|X - median|) = 1
    # scaled_mad = 1 * 1.4826
    expected = (data - 3) / (1 * 1.4826)
    np.testing.assert_array_almost_equal(result, expected)

def test_with_nan():
    """Test scaling with NaN values."""
    data = np.array([1, 2, np.nan, 4, 5])
    result = robust_scale(data)
    # NaN values should remain NaN
    assert np.isnan(result[2])
    # Other values should be scaled normally
    valid_mask = ~np.isnan(result)
    assert not np.any(np.isnan(result[valid_mask]))

def test_constant_input():
    """Test scaling with constant input."""
    data = np.array([5, 5, 5, 5, 5])
    result = robust_scale(data)
    expected = np.zeros_like(data)
    np.testing.assert_array_equal(result, expected)

def test_empty_array():
    """Test scaling with empty array."""
    data = np.array([])
    result = robust_scale(data)
    assert len(result) == 0

def test_single_value():
    """Test scaling with single value."""
    data = np.array([42])
    result = robust_scale(data)
    expected = np.zeros_like(data)
    np.testing.assert_array_equal(result, expected)

def test_custom_quantile_range():
    """Test scaling with custom quantile range."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = robust_scale(data, quantile_range=(10, 90))
    # For this data with 10-90 quantile range:
    # median = 5.5
    # Q10 ≈ 1.9, Q90 ≈ 9.1
    # scale ≈ 7.2
    expected = (data - 5.5) / (9.1 - 1.9)
    np.testing.assert_array_almost_equal(result, expected, decimal=2)

def test_different_input_types():
    """Test scaling with different input types."""
    data_list = [1, 2, 3, 4, 5]
    result = robust_scale(data_list)
    expected = robust_scale(np.array(data_list))
    np.testing.assert_array_equal(result, expected)

def test_negative_values():
    """Test scaling with negative values."""
    data = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    result = robust_scale(data)
    assert not np.any(np.isnan(result))
    # Check that zero maps to zero after scaling
    np.testing.assert_almost_equal(result[5], 0)

def test_outliers():
    """Test scaling with outliers."""
    data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
    result = robust_scale(data)
    # Check that the outlier gets scaled but doesn't dominate the scaling
    assert result[-1] > np.max(result[:-1])
    assert not np.any(np.isnan(result))

def test_invalid_method():
    """Test that invalid method raises ValueError."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Method must be one of"):
        robust_scale(data, method='invalid')

def test_invalid_quantile_range():
    """Test that invalid quantile range raises ValueError."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        robust_scale(data, quantile_range=(75, 25))  # Invalid order

def test_all_nan():
    """Test scaling with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    result = robust_scale(data)
    assert np.all(np.isnan(result))

def test_mostly_nan():
    """Test scaling with mostly NaN values."""
    data = np.array([np.nan, np.nan, 1, 2, np.nan])
    result = robust_scale(data)
    # NaN values should remain NaN
    assert np.all(np.isnan(result) == np.isnan(data))
    # Non-NaN values should be scaled
    valid_mask = ~np.isnan(result)
    assert not np.any(np.isnan(result[valid_mask]))

def test_floating_point_data():
    """Test scaling with floating-point data."""
    data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    result = robust_scale(data)
    assert not np.any(np.isnan(result))
    assert result.dtype == np.float64 