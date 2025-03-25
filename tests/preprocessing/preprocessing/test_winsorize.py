import numpy as np
import pytest
from pypulate.preprocessing import winsorize

def test_basic_winsorize():
    """Test basic winsorization."""
    data = np.array([1, 2, 3, 100, 4, 5])
    result = winsorize(data, limits=0.1)
    # With 6 values, 10% is less than 1 value, so the limits are interpolated
    assert np.allclose(result[3], 52.5)  # 100 should be capped at the 90th percentile
    assert np.allclose(result[0], 1.5)   # 1 should be raised to the 10th percentile
    assert np.allclose(result[[1,2,4,5]], [2,3,4,5])

def test_symmetric_limits():
    """Test symmetric limits."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = winsorize(data, limits=0.2)
    # With 10 values, 20% is 2 values, values are interpolated
    assert np.allclose(result[0:2], 2.8)  # Bottom 20% should be set to interpolated value
    assert np.allclose(result[-2:], 8.2)  # Top 20% should be set to interpolated value

def test_asymmetric_limits():
    """Test asymmetric limits with different upper and lower bounds."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = winsorize(data, limits=(0.1, 0.2))
    # With 10 values, 10% is 1 value for lower, 20% is 2 values for upper
    assert np.allclose(result[0], 1.9)  # Bottom 10% should be interpolated
    assert np.allclose(result[-2:], 8.2)  # Top 20% should be interpolated

def test_zero_limits():
    """Test with zero limits (no winsorization)."""
    data = np.array([1, 2, 3, 4, 5])
    result = winsorize(data, limits=0.0)
    np.testing.assert_array_equal(result, data)

def test_extreme_limits():
    """Test with extreme limits."""
    data = np.array([1, 2, 3, 4, 5])
    result = winsorize(data, limits=0.4)
    # With 40% limits, values are interpolated between 2.6 and 3.4
    assert np.allclose(result[0:2], 2.6)  # Bottom 40% interpolated
    assert np.allclose(result[3:], 3.4)   # Top 40% interpolated
    assert np.allclose(result[2], 3.0)    # Middle value unchanged

def test_with_nan():
    """Test with NaN values."""
    data = np.array([1, np.nan, 3, 100, 4, 5])
    result = winsorize(data, limits=0.1)
    assert np.isnan(result[1])  # NaN should remain NaN
    assert np.allclose(result[3], 62.0)  # 100 should be capped at interpolated value
    assert np.allclose(result[0], 1.8)   # 1 should be raised to the 10th percentile
    assert np.allclose(result[[2,4,5]], [3,4,5])

def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    result = winsorize(data, limits=0.1)
    assert len(result) == 0

def test_single_value():
    """Test with single value."""
    data = np.array([1.0])
    result = winsorize(data, limits=0.1)
    assert result[0] == 1.0

def test_all_same_value():
    """Test when all values are the same."""
    data = np.array([1.0, 1.0, 1.0, 1.0])
    result = winsorize(data, limits=0.1)
    np.testing.assert_array_equal(result, data)

def test_invalid_limits():
    """Test invalid limits raise ValueError."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        winsorize(data, limits=1.5)  # > 1.0
    with pytest.raises(ValueError):
        winsorize(data, limits=-0.1)  # < 0.0

def test_different_input_types():
    """Test with different input types."""
    data_list = [1, 2, 3, 100, 4, 5]
    data_tuple = (1, 2, 3, 100, 4, 5)
    for data in [data_list, data_tuple]:
        result = winsorize(data, limits=0.1)
        assert np.allclose(result[3], 52.5)  # 100 should be capped at interpolated value
        assert np.allclose(result[0], 1.5)   # 1 should be raised to the 10th percentile
        assert np.allclose(result[[1,2,4,5]], [2,3,4,5])

def test_floating_point_data():
    """Test with floating point data."""
    data = np.array([1.1, 2.2, 3.3, 100.5, 4.4, 5.5])
    result = winsorize(data, limits=0.1)
    assert np.allclose(result[3], 53.0)  # 100.5 should be capped at interpolated value
    assert np.allclose(result[0], 1.65)  # 1.1 should be raised to the 10th percentile
    assert np.allclose(result[[1,2,4,5]], [2.2,3.3,4.4,5.5])

def test_all_nan():
    """Test with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan])
    result = winsorize(data, limits=0.1)
    assert np.all(np.isnan(result))

def test_mostly_nan():
    """Test with mostly NaN values."""
    data = np.array([np.nan, np.nan, 1.0, 100.0, np.nan])
    result = winsorize(data, limits=0.1)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[4])
    assert np.allclose(result[2], 10.9)  # 1.0 should be interpolated
    assert np.allclose(result[3], 90.1)  # 100.0 should be interpolated 