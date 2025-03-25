import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import discretize


def test_uniform_binning():
    """Test uniform binning strategy."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = discretize(data, n_bins=5, strategy='uniform')
    
    assert len(result) == len(data)
    assert np.min(result[~np.isnan(result)]) == 1  # Minimum bin label is 1
    assert np.max(result[~np.isnan(result)]) == 5  # Maximum bin label is n_bins
    # Check if bins are roughly equal width
    unique_bins = np.unique(result)
    bin_counts = [np.sum(result == bin_val) for bin_val in unique_bins]
    assert max(bin_counts) - min(bin_counts) <= 1


def test_quantile_binning():
    """Test quantile binning strategy."""
    data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 5])
    result = discretize(data, n_bins=4, strategy='quantile')
    
    assert len(result) == len(data)
    assert np.min(result[~np.isnan(result)]) == 1
    assert np.max(result[~np.isnan(result)]) == 4
    # Check if bins have roughly equal number of samples
    unique_bins = np.unique(result)
    bin_counts = [np.sum(result == bin_val) for bin_val in unique_bins]
    assert max(bin_counts) - min(bin_counts) <= 1


def test_kmeans_binning():
    """Test k-means binning strategy."""
    data = np.array([1, 1, 8, 8, 15, 15])
    result = discretize(data, n_bins=3, strategy='kmeans')
    
    assert len(result) == len(data)
    assert np.min(result[~np.isnan(result)]) == 1
    assert np.max(result[~np.isnan(result)]) == 3
    # Check if similar values are in the same bin
    assert result[0] == result[1]  # First two values should be in same bin
    assert result[2] == result[3]  # Middle two values should be in same bin
    assert result[4] == result[5]  # Last two values should be in same bin


def test_nan_handling():
    """Test handling of NaN values."""
    data = np.array([1, np.nan, 3, 4, np.nan, 6])
    result = discretize(data, n_bins=3, strategy='uniform')
    
    assert len(result) == len(data)
    assert np.isnan(result[1])  # NaN preserved at index 1
    assert np.isnan(result[4])  # NaN preserved at index 4
    assert not np.isnan(result[0])  # Non-NaN value discretized
    assert not np.isnan(result[2])  # Non-NaN value discretized


def test_constant_data():
    """Test discretization of constant data."""
    data = np.array([5, 5, 5, 5])
    result = discretize(data, n_bins=3, strategy='uniform')
    
    assert len(result) == len(data)
    assert np.all(result == 1)  # All values should be in bin 1


def test_single_value():
    """Test discretization of single value."""
    data = np.array([3])
    result = discretize(data, n_bins=3, strategy='uniform')
    
    assert len(result) == 1
    assert result[0] == 1


def test_empty_array():
    """Test discretization of empty array."""
    data = np.array([])
    result = discretize(data, n_bins=3, strategy='uniform')
    
    assert len(result) == 0


def test_invalid_strategy():
    """Test error handling for invalid strategy."""
    data = np.array([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError):
        discretize(data, n_bins=3, strategy='invalid')


def test_different_input_types():
    """Test discretization with different input types."""
    # List input
    list_input = [1, 2, 3, 4, 5]
    list_result = discretize(list_input, n_bins=3, strategy='uniform')
    assert isinstance(list_result, np.ndarray)
    
    # Integer array
    int_input = np.array([1, 2, 3, 4, 5], dtype=int)
    int_result = discretize(int_input, n_bins=3, strategy='uniform')
    assert isinstance(int_result, np.ndarray)


def test_edge_cases():
    """Test edge cases in discretization."""
    # Very small values
    small_data = np.array([1e-6, 2e-6, 3e-6])
    small_result = discretize(small_data, n_bins=3, strategy='uniform')
    assert len(np.unique(small_result[~np.isnan(small_result)])) <= 3
    
    # Very large values
    large_data = np.array([1e6, 2e6, 3e6])
    large_result = discretize(large_data, n_bins=3, strategy='uniform')
    assert len(np.unique(large_result[~np.isnan(large_result)])) <= 3
    
    # Mixed scales
    mixed_data = np.array([1e-6, 1, 1e6])
    mixed_result = discretize(mixed_data, n_bins=3, strategy='uniform')
    assert len(np.unique(mixed_result[~np.isnan(mixed_result)])) <= 3


def test_strategies_comparison():
    """Test and compare different binning strategies."""
    data = np.array([1, 1, 1, 2, 2, 3, 4, 5, 5, 5])
    
    uniform_result = discretize(data, n_bins=3, strategy='uniform')
    quantile_result = discretize(data, n_bins=3, strategy='quantile')
    kmeans_result = discretize(data, n_bins=3, strategy='kmeans')
    
    # All strategies should produce valid bin labels
    assert np.all((uniform_result >= 1) & (uniform_result <= 3))
    assert np.all((quantile_result >= 1) & (quantile_result <= 3))
    assert np.all((kmeans_result >= 1) & (kmeans_result <= 3))
    
    # Results should be different due to different strategies
    assert not np.array_equal(uniform_result, quantile_result) or \
           not np.array_equal(quantile_result, kmeans_result) or \
           not np.array_equal(uniform_result, kmeans_result)