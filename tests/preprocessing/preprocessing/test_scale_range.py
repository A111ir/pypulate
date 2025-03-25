import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import scale_to_range


def test_basic_scaling():
    """Test basic scaling functionality."""
    data = np.array([1, 2, 3, 4, 5])
    result = scale_to_range(data, feature_range=(0, 1))
    
    assert len(result) == len(data)
    assert np.isclose(np.min(result), 0)  # Minimum should be scaled to 0
    assert np.isclose(np.max(result), 1)  # Maximum should be scaled to 1
    assert np.allclose(result, [0, 0.25, 0.5, 0.75, 1])  # Check intermediate values


def test_custom_range():
    """Test scaling to custom range."""
    data = np.array([1, 2, 3, 4, 5])
    result = scale_to_range(data, feature_range=(-1, 1))
    
    assert np.isclose(np.min(result), -1)
    assert np.isclose(np.max(result), 1)
    assert np.allclose(result, [-1, -0.5, 0, 0.5, 1])


def test_negative_range():
    """Test scaling to negative range."""
    data = np.array([1, 2, 3, 4, 5])
    result = scale_to_range(data, feature_range=(-2, -1))
    
    assert np.isclose(np.min(result), -2)
    assert np.isclose(np.max(result), -1)
    assert np.all(result < 0)  # All values should be negative


def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = scale_to_range(data, feature_range=(0, 1))
    
    assert len(result) == len(data)
    assert np.isnan(result[1])  # NaN preserved at original position
    assert np.isnan(result[4])  # NaN preserved at original position
    assert not np.isnan(result[0])  # Non-NaN values processed
    assert not np.isnan(result[-1])
    assert np.isclose(np.nanmin(result), 0)  # Min of non-NaN values
    assert np.isclose(np.nanmax(result), 1)  # Max of non-NaN values


def test_constant_data():
    """Test scaling with constant data."""
    data = np.array([5, 5, 5, 5])
    result = scale_to_range(data, feature_range=(0, 1))
    
    assert np.all(result == 0)  # All values should be mapped to feature_range[0]


def test_single_value():
    """Test scaling with single value."""
    data = np.array([3])
    result = scale_to_range(data, feature_range=(0, 1))
    
    assert len(result) == 1
    assert result[0] == 0  # Should be mapped to feature_range[0]


def test_empty_array():
    """Test scaling with empty array."""
    data = np.array([])
    result = scale_to_range(data, feature_range=(0, 1))
    
    assert len(result) == 0


def test_different_input_types():
    """Test scaling with different input types."""
    # List input
    list_input = [1, 2, 3, 4, 5]
    list_result = scale_to_range(list_input, feature_range=(0, 1))
    assert isinstance(list_result, np.ndarray)
    
    # Integer array
    int_input = np.array([1, 2, 3, 4, 5], dtype=int)
    int_result = scale_to_range(int_input, feature_range=(0, 1))
    assert isinstance(int_result, np.ndarray)
    assert int_result.dtype == np.float64


def test_edge_cases():
    """Test edge cases in scaling."""
    # Very small values
    small_data = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
    small_result = scale_to_range(small_data, feature_range=(0, 1))
    assert np.isclose(np.min(small_result), 0)
    assert np.isclose(np.max(small_result), 1)
    
    # Very large values
    large_data = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    large_result = scale_to_range(large_data, feature_range=(0, 1))
    assert np.isclose(np.min(large_result), 0)
    assert np.isclose(np.max(large_result), 1)
    
    # Mixed scales
    mixed_data = np.array([1e-6, 1e-3, 1, 1e3, 1e6])
    mixed_result = scale_to_range(mixed_data, feature_range=(0, 1))
    assert np.isclose(np.min(mixed_result), 0)
    assert np.isclose(np.max(mixed_result), 1)


def test_preserve_relative_distances():
    """Test that relative distances between values are preserved."""
    data = np.array([1, 4, 7, 10])
    result = scale_to_range(data, feature_range=(0, 1))
    
    # Check if ratios of distances are preserved
    original_ratios = np.diff(data)
    scaled_ratios = np.diff(result)
    assert np.allclose(scaled_ratios / scaled_ratios[0], original_ratios / original_ratios[0])


def test_inverse_scaling():
    """Test scaling followed by inverse scaling recovers original values."""
    data = np.array([1, 2, 3, 4, 5])
    scaled = scale_to_range(data, feature_range=(0, 1))
    
    # Calculate parameters for inverse scaling
    min_val = np.min(data)
    max_val = np.max(data)
    inverse_range = (min_val, max_val)
    
    # Apply inverse scaling
    recovered = scale_to_range(scaled, feature_range=inverse_range)
    assert np.allclose(recovered, data)


def test_invalid_range():
    """Test error handling for invalid feature range."""
    data = np.array([1, 2, 3, 4, 5])
    
    # Test min > max
    with pytest.raises(ValueError):
        scale_to_range(data, feature_range=(1, 0))
    
    # Test equal min and max
    with pytest.raises(ValueError):
        scale_to_range(data, feature_range=(1, 1)) 