import numpy as np
import pytest
from pypulate.preprocessing import normalize

def test_l1_normalization():
    """Test basic L1 normalization."""
    data = [1, 2, 3, 4]
    result = normalize(data, method='l1')
    assert np.allclose(np.sum(np.abs(result)), 1.0)
    assert np.allclose(result, [0.1, 0.2, 0.3, 0.4])

def test_l2_normalization():
    """Test basic L2 normalization."""
    data = [1, 2, 3, 4]
    result = normalize(data, method='l2')
    assert np.allclose(np.sqrt(np.sum(result**2)), 1.0)
    expected = np.array([1, 2, 3, 4]) / np.sqrt(30)
    assert np.allclose(result, expected)

def test_with_nan():
    """Test normalization with NaN values."""
    data = [1, 2, np.nan, 4]
    # L1 normalization
    result_l1 = normalize(data, method='l1')
    valid_data_l1 = result_l1[~np.isnan(result_l1)]
    assert np.allclose(np.sum(np.abs(valid_data_l1)), 1.0)
    assert np.isnan(result_l1[2])
    
    # L2 normalization
    result_l2 = normalize(data, method='l2')
    valid_data_l2 = result_l2[~np.isnan(result_l2)]
    assert np.allclose(np.sqrt(np.sum(valid_data_l2**2)), 1.0)
    assert np.isnan(result_l2[2])

def test_zero_array():
    """Test normalization with zero array."""
    data = [0, 0, 0, 0]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.all(result_l1 == 0)
    assert np.all(result_l2 == 0)

def test_empty_array():
    """Test normalization with empty array."""
    data = []
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert len(result_l1) == 0
    assert len(result_l2) == 0

def test_single_value():
    """Test normalization with single value."""
    data = [5]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.allclose(result_l1, [1.0])
    assert np.allclose(result_l2, [1.0])

def test_all_nan():
    """Test normalization with all NaN values."""
    data = [np.nan, np.nan, np.nan]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.all(np.isnan(result_l1))
    assert np.all(np.isnan(result_l2))

def test_mixed_types():
    """Test normalization with mixed numeric types."""
    data = [1, 2.5, 3, 4.5]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.allclose(np.sum(np.abs(result_l1)), 1.0)
    assert np.allclose(np.sqrt(np.sum(result_l2**2)), 1.0)

def test_negative_values():
    """Test normalization with negative values."""
    data = [-2, -1, 0, 1, 2]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.allclose(np.sum(np.abs(result_l1)), 1.0)
    assert np.allclose(np.sqrt(np.sum(result_l2**2)), 1.0)

def test_large_values():
    """Test normalization with large values."""
    data = [1e6, 2e6, 3e6, 4e6]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.allclose(np.sum(np.abs(result_l1)), 1.0)
    assert np.allclose(np.sqrt(np.sum(result_l2**2)), 1.0)

def test_small_values():
    """Test normalization with small values."""
    data = [1e-6, 2e-6, 3e-6, 4e-6]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.allclose(np.sum(np.abs(result_l1)), 1.0)
    assert np.allclose(np.sqrt(np.sum(result_l2**2)), 1.0)

def test_mostly_nan():
    """Test normalization with mostly NaN values."""
    data = [np.nan, np.nan, 1, np.nan, 2]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    valid_data_l1 = result_l1[~np.isnan(result_l1)]
    valid_data_l2 = result_l2[~np.isnan(result_l2)]
    assert np.allclose(np.sum(np.abs(valid_data_l1)), 1.0)
    assert np.allclose(np.sqrt(np.sum(valid_data_l2**2)), 1.0)
    assert np.sum(np.isnan(result_l1)) == 3
    assert np.sum(np.isnan(result_l2)) == 3

def test_different_array_types():
    """Test normalization with different array types."""
    # List
    assert isinstance(normalize([1, 2, 3], method='l1'), np.ndarray)
    assert isinstance(normalize([1, 2, 3], method='l2'), np.ndarray)
    # Numpy array
    assert isinstance(normalize(np.array([1, 2, 3]), method='l1'), np.ndarray)
    assert isinstance(normalize(np.array([1, 2, 3]), method='l2'), np.ndarray)
    # Tuple
    assert isinstance(normalize((1, 2, 3), method='l1'), np.ndarray)
    assert isinstance(normalize((1, 2, 3), method='l2'), np.ndarray)

def test_invalid_method():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError):
        normalize([1, 2, 3], method='invalid')

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    data = [1e-10, 1e10, 1e-10, 1e10]
    result_l1 = normalize(data, method='l1')
    result_l2 = normalize(data, method='l2')
    assert np.allclose(np.sum(np.abs(result_l1)), 1.0)
    assert np.allclose(np.sqrt(np.sum(result_l2**2)), 1.0) 