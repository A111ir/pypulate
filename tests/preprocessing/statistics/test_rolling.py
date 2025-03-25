import pytest
import numpy as np
import warnings
from pypulate.preprocessing.statistics import rolling_statistics

# Test fixtures
@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    return np.random.normal(loc=0, scale=1, size=100)

@pytest.fixture
def alternating_data():
    """Generate alternating data for testing skewness"""
    return np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

@pytest.fixture
def trend_data():
    """Generate trending data for testing"""
    return np.arange(1, 101, dtype=float)

# Basic functionality tests
def test_default_statistics(sample_data):
    """Test default statistics (mean and std)"""
    result = rolling_statistics(sample_data, window=5)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'mean', 'std'}
    assert all(len(arr) == len(sample_data) for arr in result.values())
    assert all(np.isnan(arr[:4]).all() for arr in result.values())
    assert all(np.isfinite(arr[4:]).all() for arr in result.values())

def test_all_statistics(sample_data):
    """Test all available statistics"""
    stats = ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']
    result = rolling_statistics(sample_data, window=5, statistics=stats)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(stats)
    assert all(len(arr) == len(sample_data) for arr in result.values())

def test_window_size(sample_data):
    """Test different window sizes"""
    windows = [3, 5, 10, 20]
    for window in windows:
        result = rolling_statistics(sample_data, window=window)
        assert all(np.isnan(arr[:window-1]).all() for arr in result.values())
        assert all(np.isfinite(arr[window-1:]).all() for arr in result.values())

# Statistical correctness tests
def test_mean_calculation(trend_data):
    """Test rolling mean calculation"""
    result = rolling_statistics(trend_data, window=3, statistics=['mean'])
    expected_mean = np.array([np.nan, np.nan] + 
                           [np.mean(trend_data[i-2:i+1]) for i in range(2, len(trend_data))])
    np.testing.assert_allclose(result['mean'], expected_mean, rtol=1e-10)

def test_std_calculation(trend_data):
    """Test rolling standard deviation calculation"""
    result = rolling_statistics(trend_data, window=3, statistics=['std'])
    expected_std = np.array([np.nan, np.nan] + 
                          [np.std(trend_data[i-2:i+1], ddof=1) for i in range(2, len(trend_data))])
    np.testing.assert_allclose(result['std'], expected_std, rtol=1e-10)

def test_min_max_calculation(trend_data):
    """Test rolling min and max calculation"""
    result = rolling_statistics(trend_data, window=3, statistics=['min', 'max'])
    expected_min = np.array([np.nan, np.nan] + 
                          [np.min(trend_data[i-2:i+1]) for i in range(2, len(trend_data))])
    expected_max = np.array([np.nan, np.nan] + 
                          [np.max(trend_data[i-2:i+1]) for i in range(2, len(trend_data))])
    np.testing.assert_allclose(result['min'], expected_min, rtol=1e-10)
    np.testing.assert_allclose(result['max'], expected_max, rtol=1e-10)

def test_median_calculation(trend_data):
    """Test rolling median calculation"""
    result = rolling_statistics(trend_data, window=3, statistics=['median'])
    expected_median = np.array([np.nan, np.nan] + 
                             [np.median(trend_data[i-2:i+1]) for i in range(2, len(trend_data))])
    np.testing.assert_allclose(result['median'], expected_median, rtol=1e-10)

def test_skew_calculation(alternating_data):
    """Test rolling skewness calculation"""
    result = rolling_statistics(alternating_data, window=3, statistics=['skew'])
    assert not np.any(np.isnan(result['skew'][2:]))  # Values after window should be finite

def test_kurt_calculation(alternating_data):
    """Test rolling kurtosis calculation"""
    result = rolling_statistics(alternating_data, window=3, statistics=['kurt'])
    assert not np.any(np.isnan(result['kurt'][2:]))  # Values after window should be finite

# Edge cases and error handling
def test_invalid_window():
    """Test behavior with invalid window size"""
    data = np.random.normal(0, 1, 10)
    with pytest.raises(ValueError, match="window must be positive"):
        rolling_statistics(data, window=0)
    with pytest.raises(ValueError, match="window must be positive"):
        rolling_statistics(data, window=-1)

def test_invalid_statistics():
    """Test behavior with invalid statistics"""
    data = np.random.normal(0, 1, 10)
    with pytest.raises(ValueError, match="Invalid statistics:"):
        rolling_statistics(data, window=3, statistics=['invalid'])

def test_window_larger_than_data():
    """Test behavior when window is larger than data"""
    data = np.random.normal(0, 1, 5)
    result = rolling_statistics(data, window=10)
    assert all(np.isnan(arr).all() for arr in result.values())

def test_single_value():
    """Test behavior with single value"""
    data = np.array([1.0])
    result = rolling_statistics(data, window=1)
    assert np.isnan(result['mean'][0])  # Need at least 2 points for statistics

# NaN handling tests
def test_all_nan():
    """Test behavior with all NaN values"""
    data = np.full(10, np.nan)
    result = rolling_statistics(data, window=3)
    assert all(np.isnan(arr).all() for arr in result.values())

def test_some_nan():
    """Test behavior with some NaN values"""
    data = np.random.normal(0, 1, 10)
    data[::2] = np.nan  # Set every other value to NaN
    result = rolling_statistics(data, window=3)
    assert all(isinstance(arr, np.ndarray) for arr in result.values())

def test_leading_trailing_nan():
    """Test behavior with NaN values at start and end"""
    data = np.random.normal(0, 1, 10)
    data[:2] = np.nan  # Leading NaNs
    data[-2:] = np.nan  # Trailing NaNs
    result = rolling_statistics(data, window=3)
    assert all(isinstance(arr, np.ndarray) for arr in result.values())

# Input type tests
def test_list_input():
    """Test that function accepts list input"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rolling_statistics(data, window=3)
    assert all(isinstance(arr, np.ndarray) for arr in result.values())

def test_tuple_input():
    """Test that function accepts tuple input"""
    data = (1.0, 2.0, 3.0, 4.0, 5.0)
    result = rolling_statistics(data, window=3)
    assert all(isinstance(arr, np.ndarray) for arr in result.values())

# Performance test
def test_large_input():
    """Test performance with large input"""
    data = np.random.normal(0, 1, 10000)
    result = rolling_statistics(data, window=100)
    assert all(isinstance(arr, np.ndarray) for arr in result.values())
    assert all(len(arr) == 10000 for arr in result.values())
    assert all(np.isnan(arr[:99]).all() for arr in result.values())
    assert all(np.isfinite(arr[99:]).any() for arr in result.values()) 