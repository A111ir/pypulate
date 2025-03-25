import numpy as np
import pytest
from pypulate.preprocessing.statistics import correlation_matrix


def test_simple_correlation():
    """Test correlation matrix calculation with simple 2D data."""
    data = np.array([[1, 2], [2, 4], [3, 6]])  # Perfect correlation
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 0], 1.0)  # Diagonal elements should be 1
    assert np.allclose(corr[1, 1], 1.0)  # Diagonal elements should be 1
    assert np.allclose(corr[0, 1], 1.0)  # Perfect positive correlation
    assert np.allclose(corr[1, 0], 1.0)  # Matrix should be symmetric


def test_perfect_negative_correlation():
    """Test correlation matrix with perfectly negatively correlated variables."""
    data = np.array([[1, -2], [2, -4], [3, -6]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 0], 1.0)
    assert np.allclose(corr[1, 1], 1.0)
    assert np.allclose(corr[0, 1], -1.0)  # Perfect negative correlation
    assert np.allclose(corr[1, 0], -1.0)  # Matrix should be symmetric


def test_uncorrelated_data():
    """Test correlation matrix with uncorrelated variables."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    data = np.column_stack((x, y))
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 0], 1.0)
    assert np.allclose(corr[1, 1], 1.0)
    assert np.abs(corr[0, 1]) < 0.3  # Correlation should be close to 0
    assert np.abs(corr[1, 0]) < 0.3  # Matrix should be symmetric


def test_nan_handling():
    """Test handling of NaN values in the data."""
    data = np.array([[1, 2], [np.nan, 4], [3, np.nan], [4, 8]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert not np.any(np.isnan(corr))  # Result should not contain NaN
    assert np.allclose(corr[0, 0], 1.0)
    assert np.allclose(corr[1, 1], 1.0)


def test_single_observation():
    """Test behavior with single observation."""
    data = np.array([[1, 2]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr, np.eye(2))  # Identity matrix for single observation


def test_constant_column():
    """Test behavior with constant column."""
    data = np.array([[1, 2], [1, 4], [1, 6]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 0], 1.0)  # Correlation of constant with itself is 1
    assert np.allclose(corr[1, 1], 1.0)  # Non-constant column with itself is 1
    assert np.isnan(corr[0, 1])  # Correlation with constant is undefined
    assert np.isnan(corr[1, 0])  # Matrix should be symmetric


def test_large_values():
    """Test handling of large values with scaling."""
    data = np.array([[1e6, 2e6], [2e6, 4e6], [3e6, 6e6]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 1], 1.0)  # Perfect correlation regardless of scale
    assert np.allclose(corr[1, 0], 1.0)  # Matrix should be symmetric


def test_small_values():
    """Test handling of small values with scaling."""
    data = np.array([[1e-6, 2e-6], [2e-6, 4e-6], [3e-6, 6e-6]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 1], 1.0)  # Perfect correlation regardless of scale
    assert np.allclose(corr[1, 0], 1.0)  # Matrix should be symmetric


def test_mixed_scales():
    """Test handling of mixed scales in data."""
    data = np.array([[1e6, 1e-6], [2e6, 2e-6], [3e6, 3e-6]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 1], 1.0)  # Perfect correlation regardless of scale
    assert np.allclose(corr[1, 0], 1.0)  # Matrix should be symmetric


def test_invalid_input_1d():
    """Test handling of 1D input."""
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input data must be a 2D array"):
        correlation_matrix(data)


def test_all_nan_column():
    """Test handling of column with all NaN values."""
    data = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
    corr = correlation_matrix(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr[0, 0], 1.0)  # Non-NaN column correlation with itself
    assert np.allclose(corr[1, 1], 1.0)  # NaN column correlation with itself
    assert np.isnan(corr[0, 1])  # Correlation with NaN column is undefined
    assert np.isnan(corr[1, 0])  # Matrix should be symmetric


def test_three_variables():
    """Test correlation matrix with three variables."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 0.1, 100)  # Strongly correlated with x
    z = np.random.normal(0, 1, 100)  # Uncorrelated with x and y
    data = np.column_stack((x, y, z))
    corr = correlation_matrix(data)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)  # Diagonal elements should be 1
    assert corr[0, 1] > 0.9  # Strong correlation between x and y
    assert np.abs(corr[0, 2]) < 0.3  # Weak correlation between x and z
    assert np.abs(corr[1, 2]) < 0.3  # Weak correlation between y and z 