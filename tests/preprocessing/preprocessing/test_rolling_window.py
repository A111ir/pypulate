import numpy as np
import pytest
from pypulate.preprocessing.preprocessing import rolling_window

def test_basic_window():
    """Test basic rolling window creation."""
    data = np.array([1, 2, 3, 4, 5])
    result = rolling_window(data, window_size=3)
    expected = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ])
    np.testing.assert_array_equal(result, expected)

def test_step_size():
    """Test rolling window with different step size."""
    data = np.array([1, 2, 3, 4, 5, 6])
    result = rolling_window(data, window_size=2, step=2)
    expected = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    np.testing.assert_array_equal(result, expected)

def test_window_larger_than_data():
    """Test when window size is larger than data length."""
    data = np.array([1, 2, 3])
    result = rolling_window(data, window_size=4)
    expected = np.zeros((0, 4))
    np.testing.assert_array_equal(result, expected)

def test_empty_data():
    """Test with empty input array."""
    data = np.array([])
    result = rolling_window(data, window_size=2)
    expected = np.zeros((0, 2))
    np.testing.assert_array_equal(result, expected)

def test_single_value():
    """Test with single value input."""
    data = np.array([1])
    result = rolling_window(data, window_size=1)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)

def test_floating_point_data():
    """Test with floating-point data."""
    data = np.array([1.5, 2.5, 3.5, 4.5])
    result = rolling_window(data, window_size=2)
    expected = np.array([
        [1.5, 2.5],
        [2.5, 3.5],
        [3.5, 4.5]
    ])
    np.testing.assert_array_equal(result, expected)

def test_negative_window_size():
    """Test that negative window size raises ValueError."""
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="window_size and step must be positive"):
        rolling_window(data, window_size=-1)

def test_negative_step():
    """Test that negative step raises ValueError."""
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="window_size and step must be positive"):
        rolling_window(data, window_size=2, step=-1)

def test_zero_window_size():
    """Test that zero window size raises ValueError."""
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="window_size and step must be positive"):
        rolling_window(data, window_size=0)

def test_zero_step():
    """Test that zero step raises ValueError."""
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="window_size and step must be positive"):
        rolling_window(data, window_size=2, step=0)

def test_different_input_types():
    """Test with different input types."""
    data = [1, 2, 3, 4]  # list input
    result = rolling_window(data, window_size=2)
    expected = np.array([
        [1, 2],
        [2, 3],
        [3, 4]
    ])
    np.testing.assert_array_equal(result, expected)

def test_large_step():
    """Test with step size larger than data length."""
    data = np.array([1, 2, 3])
    result = rolling_window(data, window_size=2, step=4)
    expected = np.array([[1, 2]])
    np.testing.assert_array_equal(result, expected)

def test_exact_window_size():
    """Test when window size equals data length."""
    data = np.array([1, 2, 3])
    result = rolling_window(data, window_size=3)
    expected = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(result, expected)

def test_step_equals_window():
    """Test when step size equals window size."""
    data = np.array([1, 2, 3, 4, 5, 6])
    result = rolling_window(data, window_size=3, step=3)
    expected = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    np.testing.assert_array_equal(result, expected) 