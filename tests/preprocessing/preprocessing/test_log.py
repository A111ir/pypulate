import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import log_transform


def test_natural_log():
    """Test natural logarithm transformation."""
    data = np.array([1, 2, 3, 4, 5])
    result = log_transform(data)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, np.log(data))


def test_base_10_log():
    """Test base 10 logarithm transformation."""
    data = np.array([1, 10, 100, 1000])
    result = log_transform(data, base=10)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, [0, 1, 2, 3])


def test_base_2_log():
    """Test base 2 logarithm transformation."""
    data = np.array([1, 2, 4, 8, 16])
    result = log_transform(data, base=2)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, [0, 1, 2, 3, 4])


def test_custom_base_log():
    """Test logarithm with custom base."""
    data = np.array([1, 3, 9, 27])
    result = log_transform(data, base=3)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, [0, 1, 2, 3])


def test_offset():
    """Test logarithm with offset for non-positive values."""
    data = np.array([0, 1, 2, 3])
    result = log_transform(data, offset=1)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, np.log([1, 2, 3, 4]))


def test_negative_values_error():
    """Test error handling for negative values without sufficient offset."""
    data = np.array([-1, 0, 1, 2])
    
    with pytest.raises(ValueError, match="Data contains non-positive values. Use a larger offset."):
        log_transform(data)


def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = log_transform(data)
    
    assert len(result) == len(data)
    assert np.isnan(result[1])  # NaN preserved at original position
    assert np.isnan(result[4])  # NaN preserved at original position
    assert not np.isnan(result[0])  # Non-NaN values processed
    assert not np.isnan(result[-1])


def test_different_input_types():
    """Test with different input types."""
    # List input
    list_input = [1, 2, 3, 4, 5]
    list_result = log_transform(list_input)
    assert isinstance(list_result, np.ndarray)
    
    # Integer array
    int_input = np.array([1, 2, 3, 4, 5], dtype=int)
    int_result = log_transform(int_input)
    assert isinstance(int_result, np.ndarray)
    assert int_result.dtype == np.float64


def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    result = log_transform(data)
    
    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_single_value():
    """Test with single value."""
    data = np.array([3])
    result = log_transform(data)
    
    assert len(result) == 1
    assert np.isfinite(result[0])
    assert np.isclose(result[0], np.log(3))


def test_large_values():
    """Test with very large values."""
    data = np.array([1e10, 1e20, 1e30])
    result = log_transform(data)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, np.log(data))


def test_small_values():
    """Test with very small positive values."""
    data = np.array([1e-10, 1e-20, 1e-30])
    result = log_transform(data)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, np.log(data))


def test_offset_with_negative_values():
    """Test offset with negative values."""
    data = np.array([-5, -3, -1, 0, 1, 3, 5])
    offset = 6  # Ensures all values become positive
    result = log_transform(data, offset=offset)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, np.log(data + offset)) 