"""Tests for technical volatility indicators."""

import pytest
import numpy as np
from pypulate.technical.volatility import (
    historical_volatility,
    atr,
    bollinger_bands,
    keltner_channels,
    donchian_channels,
    volatility_ratio
)


def test_historical_volatility():
    """Test historical volatility calculation."""
    data = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106]
    result = historical_volatility(data, period=5, annualization_factor=252)
    
    # Test volatility calculation
    assert not np.isnan(result[-1])
    assert result[-1] > 0 
    
    # Test with different period
    result_short = historical_volatility(data, period=3)
    assert not np.isnan(result_short[-1])
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_atr():
    """Test Average True Range calculation."""
    close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42]
    high = [44.55, 44.77, 44.90, 44.90, 44.85, 45.02, 45.45, 45.90]
    low = [44.12, 44.01, 44.00, 43.50, 43.98, 44.25, 44.75, 45.15]
    
    result = atr(close, high, low, period=5)
    
    # Test ATR calculation
    assert not np.isnan(result[-1])
    assert result[-1] > 0 
    
    # Test with default parameters
    result_def = atr(close)
    assert not np.isnan(result_def[-1])
    
    # Test first value is NaN
    assert np.isnan(result[0])


def test_bollinger_bands():
    """Test Bollinger Bands calculation."""
    data = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    
    upper, middle, lower = bollinger_bands(data, period=5, std_dev=2.0)
    
    # Test bands calculation
    assert not np.isnan(upper[-1])
    assert not np.isnan(middle[-1])
    assert not np.isnan(lower[-1])
    
    # Test band relationships
    assert upper[-1] > middle[-1] 
    assert middle[-1] > lower[-1] 
    
    # Test band spacing
    upper_spacing = upper[-1] - middle[-1]
    lower_spacing = middle[-1] - lower[-1]
    assert pytest.approx(upper_spacing, rel=1e-10) == lower_spacing  
    
    # Test first values are NaN
    assert np.isnan(upper[0])
    assert np.isnan(middle[0])
    assert np.isnan(lower[0])


def test_keltner_channels():
    """Test Keltner Channels calculation."""
    close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42]
    high = [44.55, 44.77, 44.90, 44.90, 44.85, 45.02, 45.45, 45.90]
    low = [44.12, 44.01, 44.00, 43.50, 43.98, 44.25, 44.75, 45.15]
    
    upper, middle, lower = keltner_channels(
        close, high, low, period=5, atr_period=5, multiplier=2.0
    )
    
    # Test channels calculation
    assert not np.isnan(upper[-1])
    assert not np.isnan(middle[-1])
    assert not np.isnan(lower[-1])
    
    # Test channel relationships
    assert upper[-1] > middle[-1] 
    assert middle[-1] > lower[-1]
    



def test_donchian_channels():
    """Test Donchian Channels calculation."""
    data = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42]
    high = [44.55, 44.77, 44.90, 44.90, 44.85, 45.02, 45.45, 45.90]
    low = [44.12, 44.01, 44.00, 43.50, 43.98, 44.25, 44.75, 45.15]
    
    upper, middle, lower = donchian_channels(data, high, low, period=5)
    
    # Test channels calculation
    assert not np.isnan(upper[-1])
    assert not np.isnan(middle[-1])
    assert not np.isnan(lower[-1])
    
    # Test channel relationships
    assert middle[-1] >= lower[-1]  
    
    # Test middle channel is average of upper and lower
    assert middle[-1] == pytest.approx((upper[-1] + lower[-1]) / 2)
    


def test_volatility_ratio():
    """Test Volatility Ratio calculation."""
    data = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106,
            105, 107, 106, 108, 107, 109, 108, 110, 109, 111]
    
    result = volatility_ratio(data, period=10, smooth_period=5)
    
    # Test ratio calculation
    assert not np.isnan(result[-1])
    assert result[-1] > 0 
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
