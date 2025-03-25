import numpy as np
import pytest
from pypulate.preprocessing import standardize

def test_basic_standardization():
    """Test basic standardization with simple array."""
    data = [1, 2, 3, 4, 5]
    result = standardize(data)
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)

def test_with_nan():
    """Test standardization with NaN values."""
    data = [1, 2, np.nan, 4, 5]
    result = standardize(data)
    valid_data = result[~np.isnan(result)]
    assert np.allclose(np.mean(valid_data), 0, atol=1e-10)
    assert np.allclose(np.std(valid_data), 1, atol=1e-10)
    assert np.isnan(result[2])

def test_constant_input():
    """Test standardization with constant input."""
    data = [3, 3, 3, 3]
    result = standardize(data)
    assert np.all(result == 0)

def test_empty_array():
    """Test standardization with empty array."""
    data = []
    result = standardize(data)
    assert len(result) == 0

def test_single_value():
    """Test standardization with single value."""
    data = [42]
    result = standardize(data)
    assert np.all(result == 0)

def test_all_nan():
    """Test standardization with all NaN values."""
    data = [np.nan, np.nan, np.nan]
    result = standardize(data)
    assert np.all(np.isnan(result))

def test_mixed_types():
    """Test standardization with mixed numeric types."""
    data = [1, 2.5, 3, 4.5, 5]
    result = standardize(data)
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)

def test_negative_values():
    """Test standardization with negative values."""
    data = [-2, -1, 0, 1, 2]
    result = standardize(data)
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)

def test_large_values():
    """Test standardization with large values."""
    data = [1e6, 2e6, 3e6, 4e6, 5e6]
    result = standardize(data)
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)

def test_small_values():
    """Test standardization with small values."""
    data = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
    result = standardize(data)
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)

def test_mostly_nan():
    """Test standardization with mostly NaN values."""
    data = [np.nan, np.nan, 1, np.nan, 2]
    result = standardize(data)
    valid_data = result[~np.isnan(result)]
    assert np.allclose(np.mean(valid_data), 0, atol=1e-10)
    assert np.allclose(np.std(valid_data), 1, atol=1e-10)
    assert np.sum(np.isnan(result)) == 3

def test_different_array_types():
    """Test standardization with different array types."""
    # List
    assert isinstance(standardize([1, 2, 3]), np.ndarray)
    # Numpy array
    assert isinstance(standardize(np.array([1, 2, 3])), np.ndarray)
    # Tuple
    assert isinstance(standardize((1, 2, 3)), np.ndarray)

def test_preserve_nan_positions():
    """Test that NaN positions are preserved after standardization."""
    data = [1, np.nan, 3, np.nan, 5]
    result = standardize(data)
    assert np.isnan(result[1])
    assert np.isnan(result[3])
    assert not np.isnan(result[0])
    assert not np.isnan(result[2])
    assert not np.isnan(result[4])

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    data = [1e-10, 1e10, 1e-10, 1e10]
    result = standardize(data)
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)