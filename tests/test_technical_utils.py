"""Tests for technical utility functions."""

import pytest
import numpy as np
from pypulate.technical.utils import (
    slope,
    rolling_max,
    rolling_min,
    rolling_std,
    rolling_var,
    zscore,
    rolling_skew,
    rolling_kurtosis
)


def test_slope():
    """Test slope calculation."""
    data = [1, 2, 3, 4, 5]
    result = slope(data, period=3)
    
    # Test basic slope calculation
    assert result[2] == pytest.approx(1.0)
    assert result[3] == pytest.approx(1.0)
    assert result[4] == pytest.approx(1.0)
    
    # Test with different data pattern
    data_mixed = [1, 3, 2, 4, 3]
    result_mixed = slope(data_mixed, period=3)
    assert not np.isnan(result_mixed[-1])
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])


def test_rolling_max():
    """Test rolling maximum calculation."""
    data = [1, 3, 2, 5, 4, 6, 5, 7]
    result = rolling_max(data, period=3)
    
    # Test basic rolling max
    assert result[2] == 3  # max of [1, 3, 2]
    assert result[3] == 5  # max of [3, 2, 5]
    assert result[4] == 5  # max of [2, 5, 4]
    
    # Test with numpy array
    data_array = np.array(data)
    np_result = rolling_max(data_array, period=3)
    np.testing.assert_array_almost_equal(result, np_result)
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])


def test_rolling_min():
    """Test rolling minimum calculation."""
    data = [1, 3, 2, 5, 4, 6, 5, 7]
    result = rolling_min(data, period=3)
    
    # Test basic rolling min
    assert result[2] == 1  # min of [1, 3, 2]
    assert result[3] == 2  # min of [3, 2, 5]
    assert result[4] == 2  # min of [2, 5, 4]
    
    # Test with numpy array
    data_array = np.array(data)
    np_result = rolling_min(data_array, period=3)
    np.testing.assert_array_almost_equal(result, np_result)
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])


def test_rolling_std():
    """Test rolling standard deviation calculation."""
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    result = rolling_std(data, period=4)
    
    # Test basic rolling std
    assert result[3] == pytest.approx(1.2910, abs=1e-4)  
    assert result[4] == pytest.approx(1.2910, abs=1e-4) 
    
    # Test with numpy array
    data_array = np.array(data)
    np_result = rolling_std(data_array, period=4)
    np.testing.assert_array_almost_equal(result, np_result)
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_rolling_var():
    """Test rolling variance calculation."""
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    result = rolling_var(data, period=4)
    
    # Test basic rolling variance
    assert result[3] == pytest.approx(1.6667, abs=1e-4)  
    assert result[4] == pytest.approx(1.6667, abs=1e-4) 
    
    # Test with numpy array
    data_array = np.array(data)
    np_result = rolling_var(data_array, period=4)
    np.testing.assert_array_almost_equal(result, np_result)
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_zscore():
    """Test Z-score calculation."""
    data = [2, 4, 6, 8, 10]
    result = zscore(data, period=3)
    
    # Test basic z-score calculation
    assert not np.isnan(result[-1])
    
    # Test with different period
    result_long = zscore(data, period=4)
    assert not np.isnan(result_long[-1])
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])


def test_rolling_skew():
    """Test rolling skewness calculation."""
    data = [1, 2, 2, 3, 3, 3, 4, 4, 5] 
    result = rolling_skew(data, period=5)
    
    # Test basic skewness calculation
    assert not np.isnan(result[-1])
    
    # Test with different period
    result_short = rolling_skew(data, period=3)
    assert not np.isnan(result_short[-1])
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_rolling_kurtosis():
    """Test rolling kurtosis calculation."""
    data = [1, 2, 2, 3, 3, 3, 4, 4, 5] 
    result = rolling_kurtosis(data, period=5)
    
    # Test basic kurtosis calculation
    assert not np.isnan(result[-1])
    
    # Test with different period
    result_short = rolling_kurtosis(data, period=4)
    assert not np.isnan(result_short[-1])
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])

