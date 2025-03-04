"""
Momentum Indicators Module

This module provides functions for calculating momentum-based indicators
for financial time series analysis.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any, List, Dict

def momentum(data, period: int = 14):
    """
    Calculate momentum over a specified period.
    
    Momentum measures the amount that a price has changed over a given period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Number of periods to calculate momentum
        
    Returns
    -------
    numpy.ndarray
        Momentum values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period, n):
        result[i] = data[i] - data[i - period]
    
    return result

def roc(data, period: int = 14):
    """
    Calculate Rate of Change (ROC) over a specified period.
    
    ROC measures the percentage change in price over a given period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Number of periods to calculate ROC
        
    Returns
    -------
    numpy.ndarray
        ROC values in percentage
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period, n):
        if data[i - period] != 0: 
            result[i] = ((data[i] / data[i - period]) - 1) * 100
    
    return result

def percent_change(data, periods: int = 1):
    """
    Calculate percentage change between consecutive periods.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    periods : int, default 1
        Number of periods to calculate change over
        
    Returns
    -------
    numpy.ndarray
        Percentage change values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(periods, n):
        if data[i - periods] != 0: 
            result[i] = ((data[i] - data[i - periods]) / data[i - periods]) * 100
    
    return result

def rsi(data, period: int = 14, smoothing_type: str = 'sma'):
    """
    Calculate Relative Strength Index (RSI) over a specified period.
    
    RSI measures the speed and change of price movements, indicating
    overbought (>70) or oversold (<30) conditions.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Number of periods to calculate RSI
    smoothing_type : str, default 'sma'
        Type of smoothing to use: 'sma' (Simple Moving Average) or 
        'ema' (Exponential Moving Average)
        
    Returns
    -------
    numpy.ndarray
        RSI values (0-100)
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    deltas = np.zeros(n)
    for i in range(1, n):
        deltas[i] = data[i] - data[i-1]
    
    gains = np.copy(deltas)
    losses = np.copy(deltas)
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = np.abs(losses) 
    
    avg_gains = np.full(n, np.nan)
    avg_losses = np.full(n, np.nan)
    
    if n > period:
        avg_gains[period] = np.mean(gains[1:period+1])
        avg_losses[period] = np.mean(losses[1:period+1])
        
        if smoothing_type.lower() == 'sma':
            for i in range(period+1, n):
                avg_gains[i] = np.mean(gains[i-period+1:i+1])
                avg_losses[i] = np.mean(losses[i-period+1:i+1])
        else:
            for i in range(period+1, n):
                avg_gains[i] = ((avg_gains[i-1] * (period-1)) + gains[i]) / period
                avg_losses[i] = ((avg_losses[i-1] * (period-1)) + losses[i]) / period
        
        for i in range(period, n):
            if avg_losses[i] == 0:
                result[i] = 100 
            else:
                rs = avg_gains[i] / avg_losses[i]
                result[i] = 100 - (100 / (1 + rs))
    
    return result

def macd(data, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    fast_period : int, default 12
        Period for the fast EMA
    slow_period : int, default 26
        Period for the slow EMA
    signal_period : int, default 9
        Period for the signal line (EMA of MACD line)
        
    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing (macd_line, signal_line, histogram)
    """
    from ..moving_averages import ema
    
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)
    
    macd_line = fast_ema - slow_ema
    
    signal_line = ema(macd_line, signal_period)
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def stochastic_oscillator(close, high, low, k_period: int = 14, d_period: int = 3):
    """
    Calculate Stochastic Oscillator.
    
    The Stochastic Oscillator is a momentum indicator that shows the location of
    the close relative to the high-low range over a set number of periods.
    
    Parameters
    ----------
    close : numpy.ndarray
        Close prices
    high : numpy.ndarray, optional
        High prices. If None, assumes close contains close prices and high=low=close
    low : numpy.ndarray, optional
        Low prices. If None, assumes close contains close prices and high=low=close
    k_period : int, default 14
        Number of periods for %K
    d_period : int, default 3
        Number of periods for %D (moving average of %K)
        
    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing (%K, %D)
    """
    close = np.asarray(close)
    n = len(close)
    
    if high is None:
        high = close
    if low is None:
        low = close
        
    high = np.asarray(high)
    low = np.asarray(low)
    
    if len(high) != n or len(low) != n:
        raise ValueError("High, low, and close arrays must have the same length")
    
    k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        
        if highest_high == lowest_low:
            k[i] = 50.0  
        else:
            k[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)
    
    d = np.full(n, np.nan)
    for i in range(k_period + d_period - 2, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d

def tsi(data, long_period: int = 25, short_period: int = 13, signal_period: int = 7):
    """
    Calculate True Strength Index (TSI).
    
    TSI is a momentum oscillator that helps identify trends and reversals.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    long_period : int, default 25
        Long period for double smoothing
    short_period : int, default 13
        Short period for double smoothing
    signal_period : int, default 7
        Period for the signal line
        
    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing (tsi_line, signal_line)
    """
    from ..moving_averages import ema
    
    data = np.asarray(data)
    n = len(data)
    
    momentum_values = np.zeros(n)
    for i in range(1, n):
        momentum_values[i] = data[i] - data[i-1]
    
    smooth1 = ema(momentum_values, long_period)
    smooth2 = ema(smooth1, short_period)
    
    abs_momentum = np.abs(momentum_values)
    abs_smooth1 = ema(abs_momentum, long_period)
    abs_smooth2 = ema(abs_smooth1, short_period)
    
    tsi_line = np.full(n, np.nan)
    for i in range(long_period + short_period - 1, n):
        if abs_smooth2[i] != 0:
            tsi_line[i] = 100.0 * smooth2[i] / abs_smooth2[i]
    
    signal_line = ema(tsi_line, signal_period)
    
    return tsi_line, signal_line

def williams_r(close, high=None, low=None, period: int = 14):
    """
    Calculate Williams %R.
    
    Williams %R is a momentum indicator that measures overbought and oversold levels.
    
    Parameters
    ----------
    close : numpy.ndarray
        Close prices
    high : numpy.ndarray, optional
        High prices. If None, assumes close contains close prices and high=low=close
    low : numpy.ndarray, optional
        Low prices. If None, assumes close contains close prices and high=low=close
    period : int, default 14
        Number of periods for calculation
        
    Returns
    -------
    numpy.ndarray
        Williams %R values (-100 to 0)
    """
    close = np.asarray(close)
    n = len(close)
    result = np.full(n, np.nan)
    
    if high is None:
        high = close
    if low is None:
        low = close
        
    high = np.asarray(high)
    low = np.asarray(low)
    
    if len(high) != n or len(low) != n:
        raise ValueError("High, low, and close arrays must have the same length")
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            result[i] = -50.0  
        else:
            result[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return result

def cci(close, period: int = 20, constant: float = 0.015):
    """
    Calculate Commodity Channel Index (CCI) using close prices.
    
    CCI measures the current price level relative to an average price level over a given period.
    This version uses only close prices instead of typical prices for simplified calculation.
    
    Parameters
    ----------
    close : numpy.ndarray
        Close prices
    period : int, default 20
        Number of periods for calculation
    constant : float, default 0.015
        Scaling constant
        
    Returns
    -------
    numpy.ndarray
        CCI values
    """
    close = np.asarray(close)
    n = len(close)
    result = np.full(n, np.nan)
        
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    mean_deviation = np.full(n, np.nan)
    for i in range(period - 1, n):
        mean_deviation[i] = np.mean(np.abs(close[i - period + 1:i + 1] - sma[i]))
    
    for i in range(period - 1, n):
        if mean_deviation[i] != 0:
            result[i] = (close[i] - sma[i]) / (constant * mean_deviation[i])
    
    return result 

def adx(data, period: int = 14):
    """
    Calculate Average Directional Index (ADX).
    
    ADX is a technical indicator used to determine the strength of a trend.

    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Number of periods for calculation

    Returns
    -------
    numpy.ndarray
        ADX values
    """
    data = np.asarray(data) 
    n = len(data)
    result = np.full(n, np.nan)
    
    if n < period:
        return result

    up_moves = np.zeros(n)
    down_moves = np.zeros(n)
    
    for i in range(1, n):
        up_moves[i] = max(0, data[i] - data[i-1])
        down_moves[i] = max(0, data[i-1] - data[i])
    
    up_avg = np.full(n, np.nan)
    down_avg = np.full(n, np.nan)
    
    for i in range(period-1, n):
        up_avg[i] = np.mean(up_moves[i-period+1:i+1])
        down_avg[i] = np.mean(down_moves[i-period+1:i+1])
    
    sum_diff = np.abs(up_avg - down_avg)
    sum_avg = up_avg + down_avg
    
    adx = np.full(n, np.nan)
    for i in range(period-1, n):
        if sum_avg[i] != 0:
            adx[i] = 100 * sum_diff[i] / sum_avg[i]
    
    return adx
        