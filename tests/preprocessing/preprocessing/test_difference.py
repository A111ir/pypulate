import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import difference


def test_first_order_difference():
    """Test first-order differencing."""
    data = np.array([1, 3, 6, 10, 15])
    result = difference(data)
    
    assert len(result) == len(data) - 1
    assert np.allclose(result, [2, 3, 4, 5])  # Each value is the difference between consecutive elements


def test_second_order_difference():
    """Test second-order differencing."""
    data = np.array([1, 3, 6, 10, 15])
    result = difference(data, order=2)
    
    assert len(result) == len(data) - 2
    assert np.allclose(result, [1, 1, 1])  # Differences of differences


def test_zero_order_difference():
    """Test zero-order differencing returns original data."""
    data = np.array([1, 3, 6, 10, 15])
    result = difference(data, order=0)
    
    assert len(result) == len(data)
    assert np.allclose(result, data)


def test_negative_order():
    """Test error handling for negative order."""
    data = np.array([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError, match="Order must be non-negative"):
        difference(data, order=-1)


def test_order_too_large():
    """Test error handling for order larger than data length."""
    data = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Order cannot be larger than the length of the data"):
        difference(data, order=4)


def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = difference(data)
    
    assert len(result) == len(data) - 1
    # When taking differences with NaN values, the result is NaN
    # [1, nan, 3, 4, nan, 6] -> [nan, nan, 1, nan, nan]
    assert np.isnan(result[0])  # 1st diff: nan - 1 = nan
    assert np.isnan(result[1])  # 2nd diff: 3 - nan = nan
    assert not np.isnan(result[2])  # 3rd diff: 4 - 3 = 1
    assert np.isnan(result[3])  # 4th diff: nan - 4 = nan
    assert np.isnan(result[4])  # 5th diff: 6 - nan = nan


def test_different_input_types():
    """Test with different input types."""
    # List input
    list_input = [1, 2, 3, 4, 5]
    list_result = difference(list_input)
    assert isinstance(list_result, np.ndarray)
    assert np.allclose(list_result, [1, 1, 1, 1])
    
    # Integer array
    int_input = np.array([1, 2, 3, 4, 5], dtype=int)
    int_result = difference(int_input)
    assert isinstance(int_result, np.ndarray)
    assert int_result.dtype == np.float64
    assert np.allclose(int_result, [1, 1, 1, 1])


def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    result = difference(data)
    
    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_single_value():
    """Test with single value."""
    data = np.array([3])
    result = difference(data)
    
    assert len(result) == 0  # First difference of single value is empty
    assert isinstance(result, np.ndarray)


def test_constant_data():
    """Test with constant data."""
    data = np.array([5, 5, 5, 5, 5])
    result = difference(data)
    
    assert len(result) == len(data) - 1
    assert np.allclose(result, 0)  # All differences should be zero


def test_arithmetic_sequence():
    """Test with arithmetic sequence."""
    data = np.array([2, 4, 6, 8, 10])  # Arithmetic sequence with common difference 2
    result = difference(data)
    
    assert len(result) == len(data) - 1
    assert np.allclose(result, 2)  # All first differences should be 2


def test_geometric_sequence():
    """Test with geometric sequence."""
    data = np.array([2, 4, 8, 16, 32])  # Geometric sequence with ratio 2
    result = difference(data)
    
    assert len(result) == len(data) - 1
    assert np.allclose(result, [2, 4, 8, 16])  # Each difference increases by factor of 2


def test_multiple_orders():
    """Test multiple orders of differencing on the same data."""
    data = np.array([1, 4, 9, 16, 25])  # Square numbers
    first_diff = difference(data)
    second_diff = difference(data, order=2)
    third_diff = difference(data, order=3)
    
    assert len(first_diff) == len(data) - 1
    assert len(second_diff) == len(data) - 2
    assert len(third_diff) == len(data) - 3
    assert np.allclose(first_diff, [3, 5, 7, 9])  # First differences
    assert np.allclose(second_diff, [2, 2, 2])    # Second differences are constant
    assert np.allclose(third_diff, [0, 0])        # Third differences are zero


def test_floating_point_data():
    """Test with floating-point data."""
    data = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    result = difference(data)
    
    assert len(result) == len(data) - 1
    assert np.allclose(result, [1.2, 0.5, 1.6, 0.3])


def test_mixed_positive_negative():
    """Test with mixed positive and negative values."""
    data = np.array([-2, 1, -3, 4, -5])
    result = difference(data)
    
    assert len(result) == len(data) - 1
    assert np.allclose(result, [3, -4, 7, -9]) 