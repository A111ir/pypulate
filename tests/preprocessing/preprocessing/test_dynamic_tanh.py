import numpy as np
import pytest
from pypulate.preprocessing import dynamic_tanh

def test_basic_dynamic_tanh():
    """Test basic dynamic tanh transformation with simple array."""
    data = [1, 2, 3, 4, 5]
    result = dynamic_tanh(data)
    # Values should be scaled to the range (-1, 1)
    assert np.all(result < 1.0)
    assert np.all(result > -1.0)
    # Check output is monotonically increasing
    assert np.all(np.diff(result) > 0)

def test_with_nan():
    """Test dynamic tanh with NaN values."""
    data = [1, 2, np.nan, 4, 5]
    result = dynamic_tanh(data)
    # Check NaN values are preserved
    assert np.isnan(result[2])
    # Values should be in range (-1, 1)
    valid_data = result[~np.isnan(result)]
    assert np.all(valid_data < 1.0)
    assert np.all(valid_data > -1.0)

def test_alpha_scaling():
    """Test that alpha parameter appropriately scales the transformation."""
    data = [1, 2, 3, 4, 5]
    result1 = dynamic_tanh(data, alpha=1.0)
    result2 = dynamic_tanh(data, alpha=2.0)
    
    # Test specific values rather than using np.all
    # With alpha=1.0: [-0.96402758, -0.76159416, 0, 0.76159416, 0.96402758]
    # With alpha=2.0: [-0.76159416, -0.46211716, 0, 0.46211716, 0.76159416]
    
    # Check that the maximum absolute value is less with higher alpha
    assert np.max(np.abs(result2)) < np.max(np.abs(result1))
    
    # Check specific elements
    assert abs(result2[0]) < abs(result1[0])  # First element
    assert abs(result2[1]) < abs(result1[1])  # Second element

def test_constant_input():
    """Test dynamic tanh with constant input."""
    data = [3, 3, 3, 3]
    result = dynamic_tanh(data)
    # For constant input, output should be zeros
    assert np.allclose(result, 0.0)

def test_empty_array():
    """Test dynamic tanh with empty array."""
    data = []
    result = dynamic_tanh(data)
    assert len(result) == 0

def test_single_value():
    """Test dynamic tanh with single value."""
    data = [42]
    result = dynamic_tanh(data)
    # Single value should return 0.0
    assert np.allclose(result, 0.0)

def test_all_nan():
    """Test dynamic tanh with all NaN values."""
    data = [np.nan, np.nan, np.nan]
    result = dynamic_tanh(data)
    assert np.all(np.isnan(result))

def test_symmetry():
    """Test that symmetric input around median produces symmetric output."""
    data = [1, 2, 3, 4, 5]  # Median is 3
    result = dynamic_tanh(data)
    # tanh is an odd function, so values equidistant from median should be opposite
    assert np.isclose(result[0], -result[4], atol=1e-10)
    assert np.isclose(result[1], -result[3], atol=1e-10)
    assert np.isclose(result[2], 0.0, atol=1e-10)  # Median maps to zero 