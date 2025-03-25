import numpy as np
import pytest
from pypulate.preprocessing.statistics import covariance_matrix


def test_simple_covariance():
    """Test covariance matrix calculation with simple 2D data."""
    data = np.array([[1, 2], [2, 4], [3, 6]])  # Perfect correlation
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert np.allclose(cov[0, 0], 1.0)  # Variance of first variable
    assert np.allclose(cov[1, 1], 4.0)  # Variance of second variable
    assert np.allclose(cov[0, 1], 2.0)  # Covariance between variables
    assert np.allclose(cov[1, 0], 2.0)  # Matrix should be symmetric


def test_uncorrelated_data():
    """Test covariance matrix with uncorrelated variables."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    data = np.column_stack((x, y))
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert np.abs(cov[0, 1]) < 0.3  # Covariance should be close to 0
    assert np.abs(cov[1, 0]) < 0.3  # Matrix should be symmetric


def test_nan_handling():
    """Test handling of NaN values in the data."""
    data = np.array([[1, 2], [np.nan, 4], [3, np.nan], [4, 8]])
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert not np.any(np.isnan(cov))  # Result should not contain NaN


def test_single_observation():
    """Test behavior with single observation."""
    data = np.array([[1, 2]])
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert np.all(np.isnan(cov))  # Should return NaN matrix


def test_insufficient_data():
    """Test behavior with insufficient data for ddof."""
    data = np.array([[1, 2]])
    cov = covariance_matrix(data, ddof=2)  # Requires at least 3 observations
    assert cov.shape == (2, 2)
    assert np.all(np.isnan(cov))


def test_constant_column():
    """Test behavior with constant column."""
    data = np.array([[1, 2], [1, 4], [1, 6]])
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert np.isclose(cov[0, 0], 0.0)  # Variance of constant column should be 0
    assert np.isclose(cov[0, 1], 0.0)  # Covariance with constant should be 0
    assert np.isclose(cov[1, 0], 0.0)  # Matrix should be symmetric


def test_large_values():
    """Test handling of large values with scaling."""
    data = np.array([[1e6, 2e6], [2e6, 4e6], [3e6, 6e6]])
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert not np.any(np.isinf(cov))  # Should not have infinite values
    assert not np.any(np.isnan(cov))  # Should not have NaN values


def test_small_values():
    """Test handling of small values with scaling."""
    data = np.array([[1e-6, 2e-6], [2e-6, 4e-6], [3e-6, 6e-6]])
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert not np.any(np.isclose(cov, 0, atol=1e-20))  # Should not underflow


def test_mixed_scales():
    """Test handling of mixed scales in data."""
    data = np.array([[1e6, 1e-6], [2e6, 2e-6], [3e6, 3e-6]])
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert not np.any(np.isinf(cov))  # Should not have infinite values
    assert not np.any(np.isnan(cov))  # Should not have NaN values


def test_invalid_input_1d():
    """Test handling of 1D input."""
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Input data must be a 2D array"):
        covariance_matrix(data)


def test_different_ddof():
    """Test different delta degrees of freedom."""
    data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    cov1 = covariance_matrix(data, ddof=1)  # Default
    cov0 = covariance_matrix(data, ddof=0)  # Population covariance
    assert np.all(cov0 <= cov1)  # Sample covariance should be larger


def test_all_nan_column():
    """Test handling of column with all NaN values."""
    data = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
    cov = covariance_matrix(data)
    assert cov.shape == (2, 2)
    assert np.all(np.isnan(cov[:, 1]))  # Column with all NaN should give NaN covariances
    assert np.all(np.isnan(cov[1, :]))  # Row with all NaN should give NaN covariances 