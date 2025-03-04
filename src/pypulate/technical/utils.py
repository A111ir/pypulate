"""
KPI Utility Functions

This module provides utility functions for calculating Key Performance Indicators (KPIs)
for financial time series analysis.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any, List, Dict

def slope(data, period: int = 5):
    """
    Calculate the slope of the time series over a specified period.
    
    This function uses linear regression to calculate the slope of the line
    that best fits the data over the specified period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 5
        Number of points to use for slope calculation
        
    Returns
    -------
    numpy.ndarray
        Slope values for each point in the time series
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    x = np.arange(period)
    
    for i in range(period - 1, n):
        y = data[i - period + 1:i + 1]
        
        if np.isnan(y).any():
            continue
            
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator != 0:
            slope = numerator / denominator
            result[i] = slope
    
    return result

def diff(data, periods: int = 1):
    """
    Calculate difference between consecutive values.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    periods : int, default 1
        Number of periods to calculate difference over
        
    Returns
    -------
    numpy.ndarray
        Difference values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(periods, n):
        result[i] = data[i] - data[i - periods]
    
    return result

def rolling_max(data, period: int = 14):
    """
    Calculate rolling maximum over a specified period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Window size for rolling maximum
        
    Returns
    -------
    numpy.ndarray
        Rolling maximum values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        if not np.isnan(window).all(): 
            result[i] = np.nanmax(window)
    
    return result

def rolling_min(data, period: int = 14):
    """
    Calculate rolling minimum over a specified period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Window size for rolling minimum
        
    Returns
    -------
    numpy.ndarray
        Rolling minimum values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        if not np.isnan(window).all():  
            result[i] = np.nanmin(window)
    
    return result

def rolling_std(data, period: int = 14):
    """
    Calculate rolling standard deviation over a specified period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Window size for rolling standard deviation
        
    Returns
    -------
    numpy.ndarray
        Rolling standard deviation values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        if not np.isnan(window).all(): 
            result[i] = np.nanstd(window, ddof=1)  
    
    return result

def rolling_var(data, period: int = 14):
    """
    Calculate rolling variance over a specified period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Window size for rolling variance
        
    Returns
    -------
    numpy.ndarray
        Rolling variance values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        if not np.isnan(window).all(): 
            result[i] = np.nanvar(window, ddof=1) 
    
    return result

def rolling_skew(data, period: int = 14):
    """
    Calculate rolling skewness over a specified period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Window size for rolling skewness
        
    Returns
    -------
    numpy.ndarray
        Rolling skewness values
    """
    from scipy import stats
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        if not np.isnan(window).all() and len(window) > 2: 
            result[i] = stats.skew(window, nan_policy='omit')
    
    return result

def rolling_kurtosis(data, period: int = 14):
    """
    Calculate rolling kurtosis over a specified period.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Window size for rolling kurtosis
        
    Returns
    -------
    numpy.ndarray
        Rolling kurtosis values
    """
    from scipy import stats
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        if not np.isnan(window).all() and len(window) > 3: 
            result[i] = stats.kurtosis(window, nan_policy='omit')
    
    return result

def zscore(data, period: int = 14):
    """
    Calculate rolling Z-score over a specified period.
    
    Z-score measures how many standard deviations a data point is from the mean.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input time series data
    period : int, default 14
        Window size for Z-score calculation
        
    Returns
    -------
    numpy.ndarray
        Z-score values
    """
    data = np.asarray(data)
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = data[i - period + 1:i + 1]
        if not np.isnan(window).all():  
            mean = np.nanmean(window[:-1])  
            std = np.nanstd(window[:-1], ddof=1)  
            if std != 0:  
                result[i] = (data[i] - mean) / std
    
    return result 

def log(data: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Calculate the natural logarithm of price data.
    
    Parameters
    ----------
    data : array-like
        Input price data as list or numpy array
        
    Returns
    -------
    numpy.ndarray
        Natural logarithm of the input data. Returns NaN for any non-positive values.
    """
    data = np.asarray(data, dtype=float)
    
    result = np.full_like(data, np.nan, dtype=float)
    
    mask = data > 0
    result[mask] = np.log(data[mask])
    
    return result


def typical_price(close: Union[List[float], np.ndarray], high: Union[List[float], np.ndarray], low: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Calculate the typical price from close, high, and low prices.
    
    Parameters
    ----------
    close : numpy.ndarray
        Close prices
    high : numpy.ndarray
        High prices
    low : numpy.ndarray
        Low prices

    Returns
    -------
    numpy.ndarray
        Typical price values
    """
    close = np.asarray(close)
    high = np.asarray(high)
    low = np.asarray(low)
    return (high + low + close) / 3