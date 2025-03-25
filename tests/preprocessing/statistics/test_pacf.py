import numpy as np
import pytest
from pypulate.preprocessing.statistics import partial_autocorrelation

@pytest.fixture
def ar1_process():
    """Generate AR(1) process with known PACF pattern."""
    np.random.seed(42)
    n = 1000
    phi = 0.7  # AR coefficient
    data = np.zeros(n)
    for t in range(1, n):
        data[t] = phi * data[t-1] + np.random.normal(0, 1)
    return data

@pytest.fixture
def ar2_process():
    """Generate AR(2) process with known PACF pattern."""
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
    """Test PACF pattern for AR(1) process."""
    pacf = partial_autocorrelation(ar1_process, max_lag=5)
    assert len(pacf) == 6  # max_lag + 1 values (including lag 0)
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert abs(pacf[1]) > 0.6  # Strong first-order correlation
    assert all(abs(pacf[2:]) < 0.4)  # Weaker higher-order correlations

def test_ar2_pattern(ar2_process):
    """Test PACF pattern for AR(2) process."""
    pacf = partial_autocorrelation(ar2_process, max_lag=5)
    assert len(pacf) == 6  # max_lag + 1 values (including lag 0)
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert abs(pacf[1]) > 0.3  # Strong first-order correlation
    assert abs(pacf[2]) > 0.2  # Moderate second-order correlation
    assert all(abs(pacf[3:]) < 0.3)  # Weaker higher-order correlations

def test_white_noise(white_noise):
    """Test PACF for white noise process."""
    pacf = partial_autocorrelation(white_noise, max_lag=5)
    assert len(pacf) == 6  # max_lag + 1 values (including lag 0)
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert all(abs(pacf[1:]) < 0.1)  # All correlations should be close to zero

def test_constant_data():
    """Test PACF for constant data."""
    data = np.ones(100)
    pacf = partial_autocorrelation(data, max_lag=5)
    assert len(pacf) == 6  # max_lag + 1 values (including lag 0)
    assert all(np.isnan(pacf))  # Should return NaN for constant data

def test_insufficient_data():
    """Test PACF with insufficient data."""
    data = np.array([1])  # Single point
    pacf = partial_autocorrelation(data, max_lag=5)
    assert all(np.isnan(pacf))  # Should return NaN for single point

def test_two_points():
    """Test PACF with two data points."""
    data = np.array([1, 2])  # Two points
    pacf = partial_autocorrelation(data, max_lag=1)
    assert len(pacf) == 2  # max_lag + 1 values (including lag 0)
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert isinstance(pacf[1], float)  # Should compute correlation for lag 1

def test_nan_handling():
    """Test PACF with NaN values."""
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    pacf = partial_autocorrelation(data, max_lag=5)
    assert len(pacf) == 6  # max_lag + 1 values (including lag 0)
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert all(isinstance(x, float) for x in pacf)  # Should handle NaN values

def test_infinite_values():
    """Test PACF with infinite values."""
    data = np.array([1.0, np.inf, 3.0, -np.inf, 5.0] * 20)
    with pytest.warns(RuntimeWarning):
        pacf = partial_autocorrelation(data, max_lag=5)
        assert len(pacf) == 6  # max_lag + 1 values (including lag 0)
        assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
        assert all(np.isnan(pacf[1:]))  # Higher lags should be NaN

def test_negative_lags():
    """Test PACF with negative number of lags."""
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        partial_autocorrelation(data, max_lag=-1)

def test_zero_lags():
    """Test PACF with zero lags."""
    data = np.random.randn(100)
    with pytest.raises(IndexError):  # Current behavior
        partial_autocorrelation(data, max_lag=0)

def test_max_lag_too_large():
    """Test PACF with max_lag larger than data length."""
    data = np.random.randn(10)
    pacf = partial_autocorrelation(data, max_lag=20)
    assert len(pacf) == 10  # Should be limited to data length - 1
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1

def test_mixed_data_types():
    """Test PACF with mixed numeric types."""
    data = np.array([1, 2.5, 3, 4.5, 5] * 20)
    pacf = partial_autocorrelation(data, max_lag=5)
    assert len(pacf) == 6  # max_lag + 1 values (including lag 0)
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert all(isinstance(x, float) for x in pacf)

def test_seasonal_pattern():
    """Test PACF with seasonal pattern."""
    np.random.seed(42)
    t = np.arange(1000)
    seasonal = np.sin(2 * np.pi * t / 12)  # Period of 12
    data = seasonal + np.random.normal(0, 0.1, 1000)
    pacf = partial_autocorrelation(data, max_lag=15)
    assert len(pacf) == 16  # max_lag + 1 values (including lag 0)
    assert pacf[0] == 1.0  # Lag 0 autocorrelation is always 1
    assert abs(pacf[12]) > 0.2  # Should show correlation at lag 12 