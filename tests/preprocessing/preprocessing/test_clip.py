import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import clip_outliers


def test_basic_clipping():
    """Test basic outlier clipping functionality."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = clip_outliers(data, lower_percentile=10, upper_percentile=90)
    
    assert len(result) == len(data)
    assert np.min(result) >= np.percentile(data, 10)
    assert np.max(result) <= np.percentile(data, 90)
    assert np.array_equal(result[2:8], data[2:8])  # Middle values unchanged


def test_default_percentiles():
    """Test clipping with default percentiles (1st and 99th)."""
    data = np.array(list(range(100)))  # 0 to 99
    result = clip_outliers(data)
    
    assert len(result) == len(data)
    assert np.min(result) >= np.percentile(data, 1)
    assert np.max(result) <= np.percentile(data, 99)
    assert np.array_equal(result[2:-2], data[2:-2])  # Most values unchanged


def test_asymmetric_percentiles():
    """Test clipping with asymmetric percentiles."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = clip_outliers(data, lower_percentile=20, upper_percentile=80)
    
    lower_bound = np.percentile(data, 20)
    upper_bound = np.percentile(data, 80)
    
    assert np.all(result >= lower_bound)
    assert np.all(result <= upper_bound)


def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6, 7, 8, 9, 10])
    result = clip_outliers(data, lower_percentile=10, upper_percentile=90)
    
    assert len(result) == len(data)
    assert np.isnan(result[1])  # NaN preserved at original position
    assert np.isnan(result[4])  # NaN preserved at original position
    assert not np.isnan(result[0])  # Non-NaN values processed
    assert not np.isnan(result[-1])


def test_extreme_percentiles():
    """Test clipping with extreme percentiles."""
    data = np.array([1, 2, 3, 4, 5])
    
    # 0th and 100th percentiles should preserve all values
    result_all = clip_outliers(data, lower_percentile=0, upper_percentile=100)
    assert np.array_equal(result_all, data)
    
    # 50th percentile both sides should clip to median
    result_median = clip_outliers(data, lower_percentile=50, upper_percentile=50)
    assert np.all(result_median == np.median(data))


def test_constant_data():
    """Test clipping with constant data."""
    data = np.array([5, 5, 5, 5, 5])
    result = clip_outliers(data, lower_percentile=10, upper_percentile=90)
    
    assert np.array_equal(result, data)  # Should remain unchanged


def test_single_value():
    """Test clipping with single value."""
    data = np.array([3])
    result = clip_outliers(data, lower_percentile=10, upper_percentile=90)
    
    assert len(result) == 1
    assert result[0] == 3  # Should remain unchanged


def test_empty_array():
    """Test clipping with empty array."""
    data = np.array([])
    result = clip_outliers(data, lower_percentile=10, upper_percentile=90)
    
    assert len(result) == 0


def test_invalid_percentiles():
    """Test error handling for invalid percentiles."""
    data = np.array([1, 2, 3, 4, 5])
    
    # Test negative percentile
    with pytest.raises(ValueError):
        clip_outliers(data, lower_percentile=-10, upper_percentile=90)
    
    # Test percentile > 100
    with pytest.raises(ValueError):
        clip_outliers(data, lower_percentile=10, upper_percentile=110)
    
    # Test lower > upper
    with pytest.raises(ValueError):
        clip_outliers(data, lower_percentile=90, upper_percentile=10)


def test_different_input_types():
    """Test clipping with different input types."""
    # List input
    list_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    list_result = clip_outliers(list_input, lower_percentile=10, upper_percentile=90)
    assert isinstance(list_result, np.ndarray)
    
    # Integer array
    int_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
    int_result = clip_outliers(int_input, lower_percentile=10, upper_percentile=90)
    assert isinstance(int_result, np.ndarray)


def test_edge_cases():
    """Test edge cases in clipping."""
    # Very small values
    small_data = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
    small_result = clip_outliers(small_data, lower_percentile=10, upper_percentile=90)
    assert len(small_result) == len(small_data)
    
    # Very large values
    large_data = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    large_result = clip_outliers(large_data, lower_percentile=10, upper_percentile=90)
    assert len(large_result) == len(large_data)
    
    # Mixed scales
    mixed_data = np.array([1e-6, 1e-3, 1, 1e3, 1e6])
    mixed_result = clip_outliers(mixed_data, lower_percentile=10, upper_percentile=90)
    assert len(mixed_result) == len(mixed_data)


def test_skewed_distribution():
    """Test clipping with skewed distribution."""
    # Right-skewed data
    right_skewed = np.array([1, 1, 2, 2, 3, 3, 4, 5, 10, 100])
    result_right = clip_outliers(right_skewed, lower_percentile=10, upper_percentile=90)
    assert np.max(result_right) < 100  # Extreme value should be clipped
    
    # Left-skewed data
    left_skewed = np.array([-100, -10, -5, -4, -3, -3, -2, -2, -1, -1])
    result_left = clip_outliers(left_skewed, lower_percentile=10, upper_percentile=90)
    assert np.min(result_left) > -100  # Extreme value should be clipped 