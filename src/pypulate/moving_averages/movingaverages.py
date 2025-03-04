"""
Moving Averages Module

This module provides various moving average implementations for financial time series analysis.
All functions use numpy arrays for input and output to ensure high performance.
"""

import numpy as np
from typing import Optional, Union, Tuple


def sma(data: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Simple Moving Average (SMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Simple moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    
    return result


def ema(data: np.ndarray, period: int = 9, alpha: Optional[float] = None) -> np.ndarray:
    """
    Exponential Moving Average (EMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
    alpha : float, optional
        Smoothing factor. If None, alpha = 2/(period+1)
        
    Returns
    -------
    numpy.ndarray
        Exponential moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data = np.asarray(data)
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    if alpha is None:
        alpha = 2 / (period + 1)
    
    # Initialize first value with SMA
    result[period - 1] = np.mean(data[:period])
    
    # Calculate EMA for remaining periods
    for i in range(period, len(data)):
        if not np.isnan(result[i - 1]):  # Only calculate if previous value exists
            result[i] = (alpha * data[i]) + ((1 - alpha) * result[i - 1])
        else:
            # If previous value is NaN, initialize with SMA of current window
            result[i] = np.mean(data[i - period + 1:i + 1])
    
    return result


def wma(data: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Weighted Moving Average (WMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Weighted moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    weights = np.arange(1, period + 1)
    weights_sum = np.sum(weights)
    
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        result[i] = np.sum(window * weights) / weights_sum
    
    return result



def tma(data: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Triangular Moving Average (TMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Triangular moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    n1 = (period + 1) // 2
    
    sma1 = sma(data, n1)
    
    result = sma(sma1, n1)
    
    return result


def smma(data: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Smoothed Moving Average (SMMA) or Running Moving Average (RMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Smoothed moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    # SMMA is equivalent to EMA with alpha = 1/period
    return ema(data, period, alpha=1/period)


def zlma(data: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Zero-Lag Moving Average (ZLMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Zero-lag moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    lag = (period - 1) // 2
    
    zero_lag_data = 2 * data - np.roll(data, lag)
    zero_lag_data[:lag] = data[:lag]
    
    result = ema(zero_lag_data, period)
    
    return result


def hma(data: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Hull Moving Average (HMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Hull moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    half_period = period // 2
    wma1 = wma(data, half_period)
    
    wma2 = wma(data, period)
    
    raw_hma = 2 * wma1 - wma2
    
    sqrt_period = int(np.sqrt(period))
    result = wma(raw_hma, sqrt_period)
    
    return result


def vwma(data: np.ndarray, volume: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Volume-Weighted Moving Average (VWMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    volume : numpy.ndarray
        Volume data corresponding to price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Volume-weighted moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if len(data) != len(volume):
        raise ValueError("Price and volume arrays must have the same length")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    for i in range(period - 1, len(data)):
        price_window = data[i - period + 1:i + 1]
        volume_window = volume[i - period + 1:i + 1]
        result[i] = np.sum(price_window * volume_window) / np.sum(volume_window)
    
    return result


def kama(data: np.ndarray, period: int = 9, fast_period: int = 2, slow_period: int = 30) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the efficiency ratio calculation
    fast_period : int, default 2
        Fast EMA period
    slow_period : int, default 30
        Slow EMA period
        
    Returns
    -------
    numpy.ndarray
        Kaufman adaptive moving average values
    """
    if period <= 0 or fast_period <= 0 or slow_period <= 0:
        raise ValueError("Periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("Fast period must be less than slow period")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    change = np.abs(data[period:] - data[:-period])
    volatility = np.zeros_like(data)
    
    for i in range(period, len(data)):
        volatility[i] = np.sum(np.abs(data[i-period+1:i+1] - data[i-period:i]))
    
    er = np.zeros_like(data)
    er[period:] = np.where(volatility[period:] != 0, change / volatility[period:], 0)
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros_like(data)
    sc[period:] = (er[period:] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    result[period - 1] = data[period - 1]
    
    for i in range(period, len(data)):
        result[i] = result[i - 1] + sc[i] * (data[i] - result[i - 1])
    
    return result


def alma(data: np.ndarray, period: int = 9, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """
    Arnaud Legoux Moving Average (ALMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
    offset : float, default 0.85
        Controls tradeoff between smoothness and responsiveness (0-1)
    sigma : float, default 6.0
        Controls the filter width
        
    Returns
    -------
    numpy.ndarray
        Arnaud Legoux moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if offset < 0 or offset > 1:
        raise ValueError("Offset must be between 0 and 1")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    m = offset * (period - 1)
    s = period / sigma
    weights = np.zeros(period)
    
    for i in range(period):
        weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
    
    weights /= np.sum(weights)
    
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        result[i] = np.sum(window * weights)
    
    return result


def frama(data: np.ndarray, period: int = 9, fc_period: int = 198) -> np.ndarray:
    """
    Fractal Adaptive Moving Average (FRAMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
    fc_period : int, default 198
        Fractal cycle period
        
    Returns
    -------
    numpy.ndarray
        Fractal adaptive moving average values
    """
    if period <= 0 or fc_period <= 0:
        raise ValueError("Periods must be positive")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    result[period - 1] = np.mean(data[:period])
    
    for i in range(period, len(data)):
        if i < period * 2:
            alpha = 2 / (period + 1)
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        else:
            n1 = period // 2
            n2 = period
            n3 = period * 2
            
            h1 = np.max(data[i-n1:i])
            l1 = np.min(data[i-n1:i])
            h2 = np.max(data[i-n2:i-n1])
            l2 = np.min(data[i-n2:i-n1])
            h3 = np.max(data[i-n3:i])
            l3 = np.min(data[i-n3:i])
            
            n1_range = h1 - l1
            n2_range = h2 - l2
            n3_range = h3 - l3
            
            if n1_range == 0 or n2_range == 0 or n3_range == 0:
                alpha = 2 / (period + 1)
            else:
                d1 = np.log(n1_range + n2_range) - np.log(n3_range)
                d2 = np.log(2)
                dimension = 1 if d2 == 0 else (d1 / d2)
                
                alpha = np.exp(-4.6 * (dimension - 1))
                alpha = max(min(alpha, 1.0), 0.01)
            
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


def jma(data: np.ndarray, period: int = 9, phase: float = 0) -> np.ndarray:
    """
    Jurik Moving Average (JMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
    phase : float, default 0
        Phase parameter (-100 to 100)
        
    Returns
    -------
    numpy.ndarray
        Jurik moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if phase < -100 or phase > 100:
        raise ValueError("Phase must be between -100 and 100")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    beta = 0.45 * (phase / 100) + 0.5
    
    alpha = 0.0962 / period + 0.5769
    power = np.exp(-3.067 * alpha)
    
    e0 = 0.0
    e1 = 0.0
    e2 = 0.0
    jma = data[0]
    result[0] = jma
    
    for i in range(1, len(data)):
        price_delta = data[i] - data[i-1]
        
        e0 = (1 - alpha) * e0 + alpha * price_delta
        
        e1 = (data[i] - jma) * power + beta * e0
        
        e2 = (1 - alpha) * e2 + alpha * e1
        
        jma += e2
        
        result[i] = jma
    
    return result


def lsma(data: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Least Squares Moving Average (LSMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Least squares moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    x = np.arange(period)
    
    for i in range(period - 1, len(data)):
        y = data[i - period + 1:i + 1]
        
        slope, intercept = np.polyfit(x, y, 1)
        
        result[i] = intercept + slope * (period - 1)
    
    return result


def mcginley_dynamic(data: np.ndarray, period: int = 9, k: float = 0.6) -> np.ndarray:
    """
    McGinley Dynamic Indicator
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
    k : float, default 0.6
        Adjustment factor
        
    Returns
    -------
    numpy.ndarray
        McGinley dynamic indicator values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    result[period - 1] = np.mean(data[:period])
    
    for i in range(period, len(data)):
        md_prev = result[i - 1]
        price = data[i]
        
        result[i] = md_prev + (price - md_prev) / (k * period * np.power(price / md_prev, 4))
    
    return result


def t3(data: np.ndarray, period: int = 9, vfactor: float = 0.7) -> np.ndarray:
    """
    Tillson T3 Moving Average
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the moving average
    vfactor : float, default 0.7
        Volume factor (0-1)
        
    Returns
    -------
    numpy.ndarray
        T3 moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if vfactor < 0 or vfactor > 1:
        raise ValueError("Volume factor must be between 0 and 1")
    
    e1 = ema(data, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)
    
    c1 = -vfactor**3
    c2 = 3 * vfactor**2 + 3 * vfactor**3
    c3 = -6 * vfactor**2 - 3 * vfactor - 3 * vfactor**3
    c4 = 1 + 3 * vfactor + vfactor**3 + 3 * vfactor**2
    
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    
    return t3


def vama(data: np.ndarray, volatility: np.ndarray, period: int = 9) -> np.ndarray:
    """
    Volatility-Adjusted Moving Average (VAMA)
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    volatility : numpy.ndarray
        Volatility data corresponding to price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    numpy.ndarray
        Volatility-adjusted moving average values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if len(data) != len(volatility):
        raise ValueError("Price and volatility arrays must have the same length")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    for i in range(period - 1, len(data)):
        price_window = data[i - period + 1:i + 1]
        vol_window = volatility[i - period + 1:i + 1]
        
        vol_sum = np.sum(vol_window)
        if vol_sum == 0:
            result[i] = np.mean(price_window)
        else:
            weights = vol_window / vol_sum
            result[i] = np.sum(price_window * weights)
    
    return result


def laguerre_filter(data: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Laguerre Filter
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    gamma : float, default 0.8
        Damping factor (0-1)
        
    Returns
    -------
    numpy.ndarray
        Laguerre filter values
    """
    if gamma < 0 or gamma > 1:
        raise ValueError("Gamma must be between 0 and 1")
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    l0 = np.zeros_like(data)
    l1 = np.zeros_like(data)
    l2 = np.zeros_like(data)
    l3 = np.zeros_like(data)
    
    for i in range(1, len(data)):
        l0[i] = (1 - gamma) * data[i] + gamma * l0[i-1]
        l1[i] = -gamma * l0[i] + l0[i-1] + gamma * l1[i-1]
        l2[i] = -gamma * l1[i] + l1[i-1] + gamma * l2[i-1]
        l3[i] = -gamma * l2[i] + l2[i-1] + gamma * l3[i-1]
        
        result[i] = (l0[i] + 2*l1[i] + 2*l2[i] + l3[i]) / 6
    
    return result


def modular_filter(data: np.ndarray, period: int = 9, phase: float = 0.5) -> np.ndarray:
    """
    Modular Filter
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    period : int, default 9
        Window size for the filter
    phase : float, default 0.5
        Phase parameter (0-1)
        
    Returns
    -------
    numpy.ndarray
        Modular filter values
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if phase < 0 or phase > 1:
        raise ValueError("Phase must be between 0 and 1")
    
    # Create output array filled with NaN
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    # Calculate alpha
    alpha = 2 / (period + 1)
    
    # Initialize filter with first price
    result[0] = data[0]
    
    # Calculate filter
    for i in range(1, len(data)):
        # Calculate filter output
        result[i] = (1 - alpha) * result[i-1] + alpha * (data[i] + phase * (data[i] - data[i-1]))
    
    return result


def rdma(data: np.ndarray) -> np.ndarray:
    """
    Rex Dog Moving Average (RDMA)
    
    This implementation follows the original RexDog definition, which is the average
    of six SMAs with periods 5, 9, 24, 50, 100, and 200.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
        
    Returns
    -------
    numpy.ndarray
        Rex Dog moving average values
    """
    sma5 = sma(data, 5)
    sma9 = sma(data, 9)
    sma24 = sma(data, 24)
    sma50 = sma(data, 50)
    sma100 = sma(data, 100)
    sma200 = sma(data, 200)
    
    result = (sma5 + sma9 + sma24 + sma50 + sma100 + sma200) / 6
    
    return result
