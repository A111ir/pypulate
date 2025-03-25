import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import polynomial_features


def test_basic_polynomial():
    """Test basic polynomial feature generation."""
    data = np.array([1, 2, 3])
    result = polynomial_features(data, degree=2)
    
    assert result.shape == (3, 3)  # 3 samples, 3 features (1, x, x²)
    assert np.allclose(result[:, 0], 1)  # Bias term
    assert np.allclose(result[:, 1], [1, 2, 3])  # Linear term
    assert np.allclose(result[:, 2], [1, 4, 9])  # Quadratic term


def test_higher_degree():
    """Test polynomial features with higher degree."""
    data = np.array([2])
    result = polynomial_features(data, degree=4)
    
    assert result.shape == (1, 5)  # 1 sample, 5 features (1, x, x², x³, x⁴)
    assert np.allclose(result[0], [1, 2, 4, 8, 16])


def test_zero_input():
    """Test polynomial features with zero input."""
    data = np.array([0])
    result = polynomial_features(data, degree=3)
    
    assert result.shape == (1, 4)
    assert np.allclose(result[0], [1, 0, 0, 0])


def test_negative_input():
    """Test polynomial features with negative input."""
    data = np.array([-2])
    result = polynomial_features(data, degree=3)
    
    assert result.shape == (1, 4)
    assert np.allclose(result[0], [1, -2, 4, -8])


def test_invalid_degree():
    """Test error handling for invalid degree."""
    data = np.array([1, 2, 3])
    
    with pytest.raises(ValueError):
        polynomial_features(data, degree=0)
    
    with pytest.raises(ValueError):
        polynomial_features(data, degree=-1)
    
    with pytest.raises(ValueError):
        polynomial_features(data, degree=1.5)


def test_nan_input():
    """Test polynomial features with NaN input."""
    data = np.array([1, np.nan, 3])
    result = polynomial_features(data, degree=2)
    
    assert result.shape == (3, 3)
    assert np.isnan(result[1, 1:]).all()  # NaN propagation
    assert np.allclose(result[0], [1, 1, 1])  # First row intact
    assert np.allclose(result[2], [1, 3, 9])  # Last row intact


def test_large_values():
    """Test polynomial features with large values."""
    data = np.array([1e3])
    result = polynomial_features(data, degree=3)
    
    assert result.shape == (1, 4)
    assert np.allclose(result[0], [1, 1e3, 1e6, 1e9])


def test_small_values():
    """Test polynomial features with small values."""
    data = np.array([1e-3])
    result = polynomial_features(data, degree=3)
    
    assert result.shape == (1, 4)
    assert np.allclose(result[0], [1, 1e-3, 1e-6, 1e-9])


def test_mixed_values():
    """Test polynomial features with mixed magnitude values."""
    data = np.array([1e-3, 1, 1e3])
    result = polynomial_features(data, degree=2)
    
    assert result.shape == (3, 3)
    expected = np.array([
        [1, 1e-3, 1e-6],
        [1, 1, 1],
        [1, 1e3, 1e6]
    ])
    assert np.allclose(result, expected)


def test_array_input():
    """Test polynomial features with different array inputs."""
    # List input
    list_input = [1, 2, 3]
    list_result = polynomial_features(list_input, degree=2)
    assert isinstance(list_result, np.ndarray)
    
    # 2D array input (should be flattened)
    array_2d = np.array([[1], [2], [3]])
    array_result = polynomial_features(array_2d, degree=2)
    assert array_result.shape == (3, 3)


def test_single_degree():
    """Test polynomial features with degree=1 (linear case)."""
    data = np.array([1, 2, 3])
    result = polynomial_features(data, degree=1)
    
    assert result.shape == (3, 2)  # Only bias and linear term
    assert np.allclose(result[:, 0], 1)  # Bias term
    assert np.allclose(result[:, 1], [1, 2, 3])  # Linear term


def test_empty_input():
    """Test polynomial features with empty input."""
    data = np.array([])
    result = polynomial_features(data, degree=2)
    
    assert result.shape == (0, 3)  # Empty but with correct number of features 