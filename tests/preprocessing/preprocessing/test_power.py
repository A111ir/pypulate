import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import power_transform


def test_yeo_johnson_positive():
    """Test Yeo-Johnson transformation with positive values."""
    data = np.array([1, 2, 3, 4, 5])
    result = power_transform(data, method='yeo-johnson')
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))  # All values should be finite
    assert np.allclose(np.mean(result), 0, atol=1e-10)  # Mean should be ~0 due to standardization
    assert np.allclose(np.std(result), 1, atol=1e-10)   # Std should be ~1 due to standardization


def test_yeo_johnson_negative():
    """Test Yeo-Johnson transformation with negative values."""
    data = np.array([-5, -4, -3, -2, -1])
    result = power_transform(data, method='yeo-johnson')
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)


def test_yeo_johnson_mixed():
    """Test Yeo-Johnson transformation with mixed positive and negative values."""
    data = np.array([-2, -1, 0, 1, 2])
    result = power_transform(data, method='yeo-johnson')
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)


def test_box_cox_positive():
    """Test Box-Cox transformation with positive values."""
    data = np.array([1, 2, 3, 4, 5])
    result = power_transform(data, method='box-cox')
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(np.mean(result), 0, atol=1e-10)
    assert np.allclose(np.std(result), 1, atol=1e-10)


def test_box_cox_negative_error():
    """Test Box-Cox transformation raises error with negative values."""
    data = np.array([-1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError, match="Box-Cox transformation requires strictly positive values"):
        power_transform(data, method='box-cox')


def test_box_cox_zero_error():
    """Test Box-Cox transformation raises error with zero values."""
    data = np.array([0, 1, 2, 3, 4])
    
    with pytest.raises(ValueError, match="Box-Cox transformation requires strictly positive values"):
        power_transform(data, method='box-cox')


def test_invalid_method():
    """Test error handling for invalid method."""
    data = np.array([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError, match="Method must be one of: 'box-cox', 'yeo-johnson'"):
        power_transform(data, method='invalid')


def test_no_standardization():
    """Test transformation without standardization."""
    data = np.array([1, 2, 3, 4, 5])
    result = power_transform(data, method='yeo-johnson', standardize=False)
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    # Result should not be standardized
    assert not np.allclose(np.mean(result), 0, atol=1e-10)
    assert not np.allclose(np.std(result), 1, atol=1e-10)


def test_different_input_types():
    """Test with different input types."""
    # List input
    list_input = [1, 2, 3, 4, 5]
    list_result = power_transform(list_input, method='yeo-johnson')
    assert isinstance(list_result, np.ndarray)
    
    # Integer array
    int_input = np.array([1, 2, 3, 4, 5], dtype=int)
    int_result = power_transform(int_input, method='yeo-johnson')
    assert isinstance(int_result, np.ndarray)
    assert int_result.dtype == np.float64


def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = power_transform(data, method='yeo-johnson')
    
    assert len(result) == len(data)
    assert np.isnan(result[1])  # NaN preserved at original position
    assert np.isnan(result[4])  # NaN preserved at original position
    assert not np.isnan(result[0])  # Non-NaN values processed
    assert not np.isnan(result[-1])


def test_constant_data():
    """Test with constant data."""
    data = np.array([5, 5, 5, 5])
    result = power_transform(data, method='yeo-johnson')
    
    assert len(result) == len(data)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, 0)  # Should be transformed to all zeros after standardization


def test_single_value():
    """Test with single value."""
    data = np.array([3])
    result = power_transform(data, method='yeo-johnson')
    
    assert len(result) == 1
    assert np.isfinite(result[0])
    assert np.isclose(result[0], 0)  # Should be transformed to 0 after standardization


def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    result = power_transform(data, method='yeo-johnson')
    
    assert len(result) == 0
    assert isinstance(result, np.ndarray) 