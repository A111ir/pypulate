"""
Volatility Measurement Functions

This module provides functions for measuring volatility in financial time series data.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any, List, Dict

def historical_volatility(data, period: int = 21, annualization_factor: int = 252):
    """
    Calculate historical volatility over a specified period.
    
    Historical volatility is the standard deviation of log returns, typically annualized.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 21
        Number of periods to calculate volatility
    annualization_factor : int, default 252
        Factor to annualize volatility (252 for daily data, 52 for weekly, 12 for monthly)
        
    Returns
    -------
    numpy.ndarray
        Historical volatility values as percentage
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    log_returns = np.full(n, np.nan)
    for i in range(1, n):
        if data[i-1] > 0 and data[i] > 0:  
            log_returns[i] = np.log(data[i] / data[i-1])
    
    for i in range(period, n):
        window = log_returns[i-period+1:i+1]
        if not np.isnan(window).all():  
            result[i] = np.nanstd(window, ddof=1) * np.sqrt(annualization_factor) * 100
    
    return result

def atr(close, high, low, period: int = 14):
    """
    Calculate Average True Range (ATR) over a specified period.
    
    ATR measures market volatility by decomposing the entire range of an asset price.
    
    Parameters
    ----------
    close : numpy.ndarray
        Close prices
    high : numpy.ndarray, optional
        High prices. If None, assumes close contains close prices and high=low=close
    low : numpy.ndarray, optional
        Low prices. If None, assumes close contains close prices and high=low=close
    period : int, default 14
        Number of periods to calculate ATR
        
    Returns
    -------
    numpy.ndarray
        ATR values
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
    
    tr = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(close[i-1]):
            continue
            

        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    for i in range(1, n):
        if i < period:
            if not np.isnan(tr[1:i+1]).all():
                result[i] = np.nanmean(tr[1:i+1])
        else:
            if not np.isnan(result[i-1]) and not np.isnan(tr[i]):
                result[i] = (result[i-1] * (period - 1) + tr[i]) / period
    
    return result

def bollinger_bands(data, period: int = 20, std_dev: float = 2.0):
    """
    Calculate Bollinger Bands over a specified period.
    
    Bollinger Bands consist of a middle band (SMA), an upper band (SMA + k*std),
    and a lower band (SMA - k*std).
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 20
        Number of periods for the moving average
    std_dev : float, default 2.0
        Number of standard deviations for the upper and lower bands
        
    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing (upper_band, middle_band, lower_band)
    """
    from ..moving_averages import sma
    from .utils import rolling_std
    
    data = np.asarray(data)
    
    middle_band = sma(data, period)
    
    std = rolling_std(data, period)
    
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    return upper_band, middle_band, lower_band

def keltner_channels(close, high, low, period: int = 20, atr_period: int = 10,
                     multiplier: float = 2.0):
    """
    Calculate Keltner Channels over a specified period.
    
    Keltner Channels consist of a middle band (EMA), an upper band (EMA + k*ATR),
    and a lower band (EMA - k*ATR).
    
    Parameters
    ----------
    close : numpy.ndarray
        Close prices
    high : numpy.ndarray, optional
        High prices. If None, assumes close contains close prices and high=low=close
    low : numpy.ndarray, optional
        Low prices. If None, assumes close contains close prices and high=low=close
    period : int, default 20
        Number of periods for the EMA
    atr_period : int, default 10
        Number of periods for the ATR
    multiplier : float, default 2.0
        Multiplier for the ATR
        
    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing (upper_channel, middle_channel, lower_channel)
    """
    from ..moving_averages import ema
    
    middle_channel = ema(close, period)
    
    atr_values = atr(close, high, low, atr_period)
    
    upper_channel = middle_channel + (multiplier * atr_values)
    lower_channel = middle_channel - (multiplier * atr_values)
    
    return upper_channel, middle_channel, lower_channel

def donchian_channels(data, high, low, period: int = 20):
    """
    Calculate Donchian Channels over a specified period.
    
    Donchian Channels consist of an upper band (highest high), a lower band (lowest low),
    and a middle band (average of upper and lower).
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data (typically close prices)
    high : numpy.ndarray, optional
        High prices. If None, uses data
    low : numpy.ndarray, optional
        Low prices. If None, uses data
    period : int, default 20
        Number of periods for the channels
        
    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing (upper_channel, middle_channel, lower_channel)
    """
    from .utils import rolling_max, rolling_min
    
    if high is None:
        high = data
    if low is None:
        low = data
        
    upper_channel = rolling_max(high, period)
    
    lower_channel = rolling_min(low, period)
    
    middle_channel = (upper_channel + lower_channel) / 2
    
    return upper_channel, middle_channel, lower_channel

def volatility_ratio(data, period: int = 21, smooth_period: int = 5):
    """
    Calculate Volatility Ratio over a specified period.
    
    Volatility Ratio compares recent volatility to historical volatility.
    Values above 1 indicate increasing volatility, values below 1 indicate decreasing volatility.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 21
        Number of periods for historical volatility
    smooth_period : int, default 5
        Number of periods to smooth the ratio
        
    Returns
    -------
    numpy.ndarray
        Volatility Ratio values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    log_returns = np.full(n, np.nan)
    for i in range(1, n):
        if data[i-1] > 0 and data[i] > 0:  
            log_returns[i] = np.log(data[i] / data[i-1])
    
    recent_vol = np.full(n, np.nan)
    for i in range(smooth_period, n):
        window = log_returns[i-smooth_period+1:i+1]
        if not np.isnan(window).all():
            recent_vol[i] = np.nanstd(window, ddof=1)
    
    hist_vol = np.full(n, np.nan)
    for i in range(period, n):
        window = log_returns[i-period+1:i+1]
        if not np.isnan(window).all():
            hist_vol[i] = np.nanstd(window, ddof=1)
    
    for i in range(period, n):
        if hist_vol[i] > 0 and not np.isnan(recent_vol[i]):
            result[i] = recent_vol[i] / hist_vol[i]
    
    return result 