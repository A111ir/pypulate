import numpy as np
import pytest
from pypulate.preprocessing import quantile_transform

def test_basic_uniform():
    """Test basic uniform quantile transformation."""
    data = np.array([1, 2, 3, 4, 5])
    result = quantile_transform(data)
    # Should map to uniform distribution in [0, 1]
    assert np.allclose(result, [0.1, 0.3, 0.5, 0.7, 0.9])

def test_basic_normal():
    """Test basic normal quantile transformation."""
    data = np.array([1, 2, 3, 4, 5])
    result = quantile_transform(data, output_distribution='normal')
    # Should map to approximately normal distribution
    assert len(result) == len(data)
    assert np.all(~np.isnan(result))
    assert np.abs(np.mean(result)) < 0.1  # Should be close to 0
    assert np.abs(np.std(result) - 1.0) < 0.15  # Should be reasonably close to 1

def test_with_nan():
    """Test with NaN values."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = quantile_transform(data)
    assert np.isnan(result[1])
    assert np.isnan(result[4])
    assert np.allclose(result[[0,2,3,5]], [0.125, 0.375, 0.625, 0.875])

def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    result = quantile_transform(data)
    assert len(result) == 0

def test_single_value():
    """Test with single value."""
    data = np.array([1.0])
    result = quantile_transform(data)
    assert result[0] == 0.5  # Single value should map to 0.5 for uniform
    
    result_normal = quantile_transform(data, output_distribution='normal')
    assert result_normal[0] == 0.0  # Single value should map to 0 for normal

def test_all_same_value():
    """Test when all values are the same."""
    data = np.array([1.0, 1.0, 1.0, 1.0])
    result = quantile_transform(data)
    assert np.allclose(result, 0.5)  # All values should map to 0.5 for uniform
    
    result_normal = quantile_transform(data, output_distribution='normal')
    assert np.allclose(result_normal, 0.0)  # All values should map to 0 for normal

def test_all_nan():
    """Test with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    result = quantile_transform(data)
    assert np.all(np.isnan(result))

def test_mostly_nan():
    """Test with mostly NaN values."""
    data = np.array([np.nan, np.nan, 1.0, 2.0, np.nan])
    result = quantile_transform(data)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[4])
    assert np.allclose(result[2:4], [0.25, 0.75])

def test_different_input_types():
    """Test with different input types."""
    data_list = [1, 2, 3, 4, 5]
    data_tuple = (1, 2, 3, 4, 5)
    for data in [data_list, data_tuple]:
        result = quantile_transform(data)
        assert np.allclose(result, [0.1, 0.3, 0.5, 0.7, 0.9])

def test_floating_point_data():
    """Test with floating point data."""
    data = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    result = quantile_transform(data)
    assert np.allclose(result, [0.1, 0.3, 0.5, 0.7, 0.9])

def test_negative_values():
    """Test with negative values."""
    data = np.array([-5, -3, 0, 3, 5])
    result = quantile_transform(data)
    assert np.allclose(result, [0.1, 0.3, 0.5, 0.7, 0.9])

def test_uneven_spacing():
    """Test with unevenly spaced data."""
    data = np.array([1, 2, 10, 100, 1000])
    result = quantile_transform(data)
    assert np.allclose(result, [0.1, 0.3, 0.5, 0.7, 0.9])

def test_custom_n_quantiles():
    """Test with custom number of quantiles."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = quantile_transform(data, n_quantiles=5)
    # Should have more coarse-grained binning
    assert len(np.unique(result)) <= 5

def test_invalid_distribution():
    """Test invalid output distribution raises ValueError."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        quantile_transform(data, output_distribution='invalid')

def test_extreme_values():
    """Test with extreme value differences."""
    data = np.array([1e-10, 1, 1e10])
    result = quantile_transform(data)
    assert np.allclose(result, [0.167, 0.5, 0.833], rtol=1e-2)

def test_repeated_values():
    """Test with repeated values."""
    data = np.array([1, 1, 2, 2, 3, 3])
    result = quantile_transform(data)
    # Repeated values should get same quantile
    assert np.allclose(result[0], result[1])
    assert np.allclose(result[2], result[3])
    assert np.allclose(result[4], result[5]) 