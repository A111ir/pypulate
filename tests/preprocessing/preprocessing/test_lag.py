import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import lag_features

def test_basic_lag():
    """Test basic lag feature creation with a single lag."""
    data = np.array([1, 2, 3, 4, 5])
    result = lag_features(data, lags=[1])
    expected = np.array([
        [1, np.nan],
        [2, 1],
        [3, 2],
        [4, 3],
        [5, 4]
    ])
    np.testing.assert_array_equal(result, expected)

def test_multiple_lags():
    """Test creating multiple lag features."""
    data = np.array([1, 2, 3, 4, 5])
    result = lag_features(data, lags=[1, 2])
    expected = np.array([
        [1, np.nan, np.nan],
        [2, 1, np.nan],
        [3, 2, 1],
        [4, 3, 2],
        [5, 4, 3]
    ])
    np.testing.assert_array_equal(result, expected)

def test_zero_lag():
    """Test that zero lag returns original data."""
    data = np.array([1, 2, 3])
    result = lag_features(data, lags=[0])
    expected = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    np.testing.assert_array_equal(result, expected)

def test_negative_lag():
    """Test that negative lags are ignored."""
    data = np.array([1, 2, 3])
    result = lag_features(data, lags=[-1])
    expected = np.array([
        [1, np.nan],
        [2, np.nan],
        [3, np.nan]
    ])
    np.testing.assert_array_equal(result, expected)

def test_empty_data():
    """Test handling of empty input array."""
    data = np.array([])
    result = lag_features(data, lags=[1])
    assert result.shape == (0, 2)

def test_single_value():
    """Test with single value input."""
    data = np.array([1])
    result = lag_features(data, lags=[1, 2])
    expected = np.array([[1, np.nan, np.nan]])
    np.testing.assert_array_equal(result, expected)

def test_nan_handling():
    """Test handling of NaN values in input."""
    data = np.array([1, np.nan, 3, 4])
    result = lag_features(data, lags=[1])
    expected = np.array([
        [1, np.nan],
        [np.nan, 1],
        [3, np.nan],
        [4, 3]
    ])
    np.testing.assert_array_equal(result, expected)

def test_different_input_types():
    """Test with different input types."""
    data = [1, 2, 3]  # list input
    result = lag_features(data, lags=[1])
    expected = np.array([
        [1, np.nan],
        [2, 1],
        [3, 2]
    ])
    np.testing.assert_array_equal(result, expected)

def test_large_lag():
    """Test with lag larger than data length."""
    data = np.array([1, 2, 3])
    result = lag_features(data, lags=[4])
    expected = np.array([
        [1, np.nan],
        [2, np.nan],
        [3, np.nan]
    ])
    np.testing.assert_array_equal(result, expected)

def test_mixed_lags():
    """Test with mixed valid and invalid lags."""
    data = np.array([1, 2, 3, 4])
    result = lag_features(data, lags=[1, -1, 2, 0])
    expected = np.array([
        [1, np.nan, np.nan, np.nan, 1],
        [2, 1, np.nan, np.nan, 2],
        [3, 2, np.nan, 1, 3],
        [4, 3, np.nan, 2, 4]
    ])
    np.testing.assert_array_equal(result, expected)

def test_floating_point_data():
    """Test with floating-point data."""
    data = np.array([1.5, 2.5, 3.5])
    result = lag_features(data, lags=[1])
    expected = np.array([
        [1.5, np.nan],
        [2.5, 1.5],
        [3.5, 2.5]
    ])
    np.testing.assert_array_equal(result, expected)

def test_empty_lags():
    """Test with empty lags list."""
    data = np.array([1, 2, 3])
    result = lag_features(data, lags=[])
    expected = np.array([[1], [2], [3]])
    np.testing.assert_array_equal(result, expected) 