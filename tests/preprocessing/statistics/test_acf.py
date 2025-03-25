import numpy as np
import pytest
from pypulate.preprocessing.statistics import autocorrelation

@pytest.fixture
def ar1_process():
    """Generate AR(1) process with known ACF pattern."""
    np.random.seed(42)
    n = 1000
    phi = 0.7  # AR coefficient
    data = np.zeros(n)
    for t in range(1, n):
        data[t] = phi * data[t-1] + np.random.normal(0, 1)
    return data

@pytest.fixture
def ar2_process():
    """Generate AR(2) process with known ACF pattern."""
    np.random.seed(42)
    n = 1000
    phi1, phi2 = 0.4, 0.3  # AR coefficients
    data = np.zeros(n)
    for t in range(2, n):
        data[t] = phi1 * data[t-1] + phi2 * data[t-2] + np.random.normal(0, 1)
    return data

@pytest.fixture
def white_noise():
    """Generate white noise process."""
    np.random.seed(42)
    return np.random.normal(0, 1, 1000)

def test_ar1_pattern(ar1_process):
    """Test ACF pattern for AR(1) process."""
    acf = autocorrelation(ar1_process, max_lag=5)
    assert len(acf) == 6  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert 0.6 < acf[1] < 0.8  # Should be close to phi=0.7
    assert all(acf[i] > acf[i+1] for i in range(1, len(acf)-1))  # Geometric decay

def test_ar2_pattern(ar2_process):
    """Test ACF pattern for AR(2) process."""
    acf = autocorrelation(ar2_process, max_lag=5)
    assert len(acf) == 6  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert acf[1] > 0.5  # Strong first-order correlation
    assert acf[2] > 0.3  # Significant second-order correlation
    assert all(acf[i] > acf[i+1] for i in range(2, len(acf)-1))  # Decay after lag 2

def test_white_noise(white_noise):
    """Test ACF for white noise process."""
    acf = autocorrelation(white_noise, max_lag=5)
    assert len(acf) == 6  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert all(abs(acf[1:]) < 0.1)  # All correlations should be close to zero

def test_constant_data():
    """Test ACF for constant data."""
    data = np.ones(100)
    acf = autocorrelation(data, max_lag=5)
    assert len(acf) == 6  # max_lag + 1 values (including lag 0)
    assert all(np.isnan(acf))  # Should return NaN for constant data

def test_insufficient_data():
    """Test ACF with insufficient data."""
    data = np.array([1])  # Single point
    acf = autocorrelation(data, max_lag=5)
    assert all(np.isnan(acf))  # Should return NaN for single point

def test_two_points():
    """Test ACF with two data points."""
    data = np.array([1, 2])  # Two points
    acf = autocorrelation(data, max_lag=1)
    assert len(acf) == 2  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert isinstance(acf[1], float)  # Should compute correlation for lag 1

def test_nan_handling():
    """Test ACF with NaN values."""
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    acf = autocorrelation(data, max_lag=5)
    assert len(acf) == 6  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert all(isinstance(x, float) for x in acf)  # Should handle NaN values

def test_infinite_values():
    """Test ACF with infinite values."""
    data = np.array([1.0, np.inf, 3.0, -np.inf, 5.0] * 20)
    with pytest.warns(RuntimeWarning):
        acf = autocorrelation(data, max_lag=5)
        assert len(acf) == 6  # max_lag + 1 values (including lag 0)
        assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
        assert all(abs(acf[1:]) <= 1.0)  # Values should be bounded by [-1, 1]

def test_negative_lags():
    """Test ACF with negative number of lags."""
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        autocorrelation(data, max_lag=-1)

def test_zero_lags():
    """Test ACF with zero lags."""
    data = np.random.randn(100)
    acf = autocorrelation(data, max_lag=0)
    assert len(acf) == 1  # Only lag 0
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1

def test_max_lag_too_large():
    """Test ACF with max_lag larger than data length."""
    data = np.random.randn(10)
    acf = autocorrelation(data, max_lag=20)
    assert len(acf) == 10  # Should be limited to data length - 1
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1

def test_mixed_data_types():
    """Test ACF with mixed numeric types."""
    data = np.array([1, 2.5, 3, 4.5, 5] * 20)
    acf = autocorrelation(data, max_lag=5)
    assert len(acf) == 6  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert all(isinstance(x, float) for x in acf)

def test_seasonal_pattern():
    """Test ACF with seasonal pattern."""
    np.random.seed(42)
    t = np.arange(1000)
    seasonal = np.sin(2 * np.pi * t / 12)  # Period of 12
    data = seasonal + np.random.normal(0, 0.1, 1000)
    acf = autocorrelation(data, max_lag=15)
    assert len(acf) == 16  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert abs(acf[12]) > 0.8  # Strong correlation at seasonal lag

def test_alternating_series():
    """Test ACF with alternating series."""
    data = np.array([1, -1] * 50)  # Perfect alternating pattern
    acf = autocorrelation(data, max_lag=5)
    assert len(acf) == 6  # max_lag + 1 values (including lag 0)
    assert acf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert np.isclose(acf[1], -1.0)  # Perfect negative correlation at lag 1
    assert np.isclose(acf[2], 1.0)  # Perfect positive correlation at lag 2 