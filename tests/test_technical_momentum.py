"""Tests for technical momentum indicators."""

import pytest
import numpy as np
from pypulate.technical.momentum import (
    momentum, roc, rsi, macd, stochastic_oscillator,
    tsi, williams_r, cci, percent_change
)


def test_momentum():
    """Test momentum calculation."""
    data = [10, 12, 13, 11, 14, 15, 16]
    result = momentum(data, period=3)
    
    # Test basic momentum calculation
    assert result[3] == pytest.approx(1)  # 11 - 10
    assert result[4] == pytest.approx(2)  # 14 - 12
    assert result[5] == pytest.approx(2)  # 15 - 13
    
    # Test with numpy array
    data_array = np.array(data)
    np_result = momentum(data_array, period=3)
    np.testing.assert_array_almost_equal(result, np_result)
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])


def test_roc():
    """Test Rate of Change calculation."""
    data = [10, 11, 12, 11, 13]
    result = roc(data, period=2)
    
    # Test basic ROC calculation
    assert result[2] == pytest.approx(20.0)  # ((12 - 10) / 10) * 100
    assert result[3] == pytest.approx(0.0)   # ((11 - 11) / 11) * 100
    assert result[4] == pytest.approx(8.33, abs=1e-2)  # ((13 - 12) / 12) * 100
    
    # Test with numpy array
    data_array = np.array(data)
    np_result = roc(data_array, period=2)
    np.testing.assert_array_almost_equal(result, np_result)
    
    # Test first values are NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])


def test_rsi():
    """Test Relative Strength Index calculation."""
    data = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    
    # Test RSI with SMA smoothing
    result_sma = rsi(data, period=14, smoothing_type='sma')
    assert result_sma[-1] == pytest.approx(70.464135, abs=1e-2)
    
    # Test RSI with EMA smoothing
    result_ema = rsi(data, period=14, smoothing_type='ema')
    assert result_ema[-1] == pytest.approx(70.464135, abs=1e-2)
    
    # Test first values are NaN
    assert np.isnan(result_sma[0])
    assert np.isnan(result_ema[0])


def test_macd():
    """Test MACD calculation."""
    data = [
        25.0, 26.0, 27.0, 28.0, 27.0, 26.0, 25.0, 24.0, 25.0, 26.0,
        27.0, 28.0, 27.0, 26.0, 25.0, 24.0, 25.0, 26.0, 27.0, 28.0
    ]
    
    macd_line, signal_line, histogram = macd(
        data, fast_period=3, slow_period=6, signal_period=3
    )
    
    # Test MACD components are calculated
    assert not np.isnan(macd_line[-1])
    assert not np.isnan(signal_line[-1])
    assert not np.isnan(histogram[-1])
    
    # Test MACD line equals fast EMA - slow EMA
    assert macd_line[-1] == pytest.approx(0.58471171, abs=1e-4)
    
    # Test histogram equals MACD line - signal line
    assert histogram[-1] == pytest.approx(
        macd_line[-1] - signal_line[-1], abs=1e-4
    )


def test_stochastic_oscillator():
    """Test Stochastic Oscillator calculation."""
    close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42]
    high = [44.55, 44.77, 44.90, 44.90, 44.85, 45.02, 45.45, 45.90]
    low = [44.12, 44.01, 44.00, 43.50, 43.98, 44.25, 44.75, 45.15]
    
    k, d = stochastic_oscillator(close, high, low, k_period=5, d_period=3)
    
    # Test %K calculation
    assert k[-1] == pytest.approx(80.00, abs=1e-2)
    
    # Test %D calculation
    assert d[-1] == pytest.approx(83.1837, abs=1e-2)
    


def test_tsi():
    """Test True Strength Index calculation."""
    data = [
        25.0, 26.0, 27.0, 28.0, 27.0, 26.0, 25.0, 24.0, 25.0, 26.0,
        27.0, 28.0, 27.0, 26.0, 25.0, 24.0, 25.0, 26.0, 27.0, 28.0
    ]
    
    tsi_line, signal_line = tsi(
        data, long_period=12, short_period=6, signal_period=3
    )
    
    # Test TSI components are calculated
    assert not np.isnan(tsi_line[-1])
    assert not np.isnan(signal_line[-1])
    
    # Test TSI values are within valid range (-100 to 100)
    assert -100 <= tsi_line[-1] <= 100
    assert -100 <= signal_line[-1] <= 100


def test_williams_r():
    """Test Williams %R calculation."""
    close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42]
    high = [44.55, 44.77, 44.90, 44.90, 44.85, 45.02, 45.45, 45.90]
    low = [44.12, 44.01, 44.00, 43.50, 43.98, 44.25, 44.75, 45.15]
    
    result = williams_r(close, high, low, period=5)
    
    # Test Williams %R calculation
    assert result[-1] == pytest.approx(-20.0, abs=1e-2)
    
    # Test values are within valid range (-100 to 0)
    assert -100 <= result[-1] <= 0
    


def test_cci():
    """Test Commodity Channel Index calculation."""
    close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42]
    high = [44.55, 44.77, 44.90, 44.90, 44.85, 45.02, 45.45, 45.90]
    low = [44.12, 44.01, 44.00, 43.50, 43.98, 44.25, 44.75, 45.15]
    
    result = cci(close, high, low, period=5)
    
    # Test CCI calculation
    assert not np.isnan(result[-1])



def test_percent_change():
    """Test percent change calculation."""
    data = [100, 102, 101, 103, 105]
    result = percent_change(data, periods=1)
    
    # Test basic percent change calculation
    assert result[1] == pytest.approx(2.0)    # ((102 - 100) / 100) * 100
    assert result[2] == pytest.approx(-0.98, abs=1e-2)  # ((101 - 102) / 102) * 100
    assert result[3] == pytest.approx(1.98, abs=1e-2)   # ((103 - 101) / 101) * 100
    
    # Test with multiple periods
    result_2 = percent_change(data, periods=2)
    assert result_2[2] == pytest.approx(1.0)  # ((101 - 100) / 100) * 100
    
    # Test first value is NaN
    assert np.isnan(result[0])
