"""
Moving Averages Module

This module provides various moving average implementations for financial time series analysis.
All functions use numpy arrays for input and output to ensure high performance.
"""

import numpy as np
from typing import Optional
from numpy.typing import ArrayLike, NDArray


def sma(data: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Simple Moving Average (SMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Simple moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    # Check if we have enough data points
    if len(data_array) < period:
        # Return array of NaNs if we don't have enough data
        return result
    
    weights = np.ones(period) / period
    convolved = np.convolve(data_array, weights, mode='valid')
    
    result[period-1:] = convolved
    
    return result


def ema(data: ArrayLike, period: int = 9, alpha: Optional[float] = None) -> NDArray[np.float64]:
    """
    Exponential Moving Average (EMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
    alpha : float, optional
        Smoothing factor. If None, alpha = 2/(period+1)
        
    Returns
    -------
    NDArray[np.float64]
        Exponential moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    # Check if we have enough data points
    if len(data_array) < period:
        # Return array of NaNs if we don't have enough data
        return result
    
    if alpha is None:
        alpha = 2 / (period + 1)
    
    result[period - 1] = np.mean(data_array[:period])
    
    if len(data_array) > period:
        for i in range(period, len(data_array)):
            result[i] = alpha * data_array[i] + (1 - alpha) * result[i - 1]
    
    return result


def wma(data: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Weighted Moving Average (WMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Weighted moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    # Check if we have enough data points
    if len(data_array) < period:
        # Return array of NaNs if we don't have enough data
        return result
    
    weights = np.arange(1, period + 1)
    weights_sum: float = np.sum(weights)
    
    weighted_data = np.convolve(data_array, weights[::-1]/weights_sum, mode='valid')
    
    result[period-1:] = weighted_data
    
    return result



def tma(data: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Triangular Moving Average (TMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Triangular moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    # Check if we have enough data points
    # TMA needs at least 2*period-1 data points (for two consecutive SMAs)
    n1 = (period + 1) // 2
    min_data_points = period + n1 - 1
    
    if len(data_array) < min_data_points:
        # Return array of NaNs if we don't have enough data
        return result
    
    sma1 = sma(data_array, n1)
    result = sma(sma1, n1)
    
    return result


def smma(data: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Smoothed Moving Average (SMMA) or Running Moving Average (RMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Smoothed moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    return ema(data, period, alpha=1/period)


def zlma(data: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Zero-Lag Moving Average (ZLMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Zero-lag moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data)
    result = np.full_like(data_array, np.float64('nan'))
    
    # Check if we have enough data points
    if len(data_array) < period:
        # Return array of NaNs if we don't have enough data
        return result
    
    lag = (period - 1) // 2
    
    zero_lag_data = 2 * data_array - np.roll(data_array, lag)
    zero_lag_data[:lag] = data_array[:lag]
    
    result = ema(zero_lag_data, period)
    
    return result


def hma(data: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Hull Moving Average (HMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Hull moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    # Check if we have enough data points
    # HMA requires enough data for the wma functions (period/2 and period)
    # and an additional sqrt(period) for the final wma
    sqrt_period = int(np.sqrt(period))
    min_data_points = period + sqrt_period - 1
    
    if len(data_array) < min_data_points:
        return result
    
    period_half = period // 2
    
    wma1 = wma(data_array, period_half)
    wma2 = wma(data_array, period)
    
    # 2 * fast WMA - slow WMA
    wma_diff = 2 * wma1 - wma2
    
    # Final Hull MA is a WMA with period sqrt(n)
    result = wma(wma_diff, sqrt_period)
    
    return result


def vwma(data: ArrayLike, volume: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Volume-Weighted Moving Average (VWMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    volume : ArrayLike
        Volume data corresponding to price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Volume-weighted moving average values

    Raises
    ------
    ValueError
        If period is not positive
        If price and volume arrays have different lengths
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    
    if len(data_array) != len(volume_array):
        raise ValueError("Price and volume arrays must have the same length")
    
    result = np.full_like(data_array, np.nan)
    
    price_volume = data_array * volume_array
    
    cum_price_volume = np.cumsum(price_volume)
    cum_volume = np.cumsum(volume_array)
    
    if len(data_array) >= period:
        rolling_price_volume = np.zeros_like(data_array)
        rolling_volume = np.zeros_like(data_array)
        
        rolling_price_volume[period-1:] = cum_price_volume[period-1:]
        rolling_volume[period-1:] = cum_volume[period-1:]
        
        if period > 1:
            rolling_price_volume[period:] -= cum_price_volume[:-period]
            rolling_volume[period:] -= cum_volume[:-period]
        
        mask = rolling_volume[period-1:] != 0
        result[period-1:][mask] = rolling_price_volume[period-1:][mask] / rolling_volume[period-1:][mask]
        
        if not np.all(mask):
            for i in range(period-1, len(data_array)):
                if rolling_volume[i] == 0:
                    result[i] = np.mean(data_array[i-period+1:i+1])
    
    return result


def kama(data: ArrayLike, period: int = 9, fast_period: int = 2, slow_period: int = 30) -> NDArray[np.float64]:
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the efficiency ratio calculation
    fast_period : int, default 2
        Fast EMA period
    slow_period : int, default 30
        Slow EMA period
        
    Returns
    -------
    NDArray[np.float64]
        Kaufman adaptive moving average values

    Raises
    ------
    ValueError
        If period is not positive
        If fast_period is not positive
        If slow_period is not positive
        If fast_period is not less than slow_period
    """
    if period <= 0 or fast_period <= 0 or slow_period <= 0:
        raise ValueError("Periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("Fast period must be less than slow period")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    direction = np.abs(np.diff(np.concatenate([[data_array[0]], data_array[period:]])))
    
    volatility = np.zeros_like(data_array)
    
    abs_diff = np.abs(np.diff(data_array))
    
    for i in range(period, len(data_array)):
        volatility[i] = np.sum(abs_diff[i-period:i])
    
    er = np.zeros_like(data_array)
    mask = volatility[period:] != 0
    er_indices = np.arange(period, len(data_array))[mask]
    er[er_indices] = direction[er_indices-period] / volatility[er_indices]
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros_like(data_array)
    sc[period:] = (er[period:] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    result[period - 1] = data_array[period - 1]
    
    for i in range(period, len(data_array)):
        result[i] = result[i - 1] + sc[i] * (data_array[i] - result[i - 1])
    
    return result


def alma(data: ArrayLike, period: int = 9, offset: float = 0.85, sigma: float = 6.0) -> NDArray[np.float64]:
    """
    Arnaud Legoux Moving Average (ALMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
    offset : float, default 0.85
        Controls tradeoff between smoothness and responsiveness (0-1)
    sigma : float, default 6.0
        Controls the filter width
        
    Returns
    -------
    NDArray[np.float64]
        Arnaud Legoux moving average values

    Raises
    ------
    ValueError
        If period is not positive
        If offset is not between 0 and 1
        If sigma is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if offset < 0 or offset > 1:
        raise ValueError("Offset must be between 0 and 1")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    m = offset * (period - 1)
    s = period / sigma
    
    indices = np.arange(period)
    weights = np.exp(-((indices - m) ** 2) / (2 * s * s))
    weights /= np.sum(weights)
    
    alma_values = np.convolve(data_array, weights[::-1], mode='valid')
    
    result[period-1:] = alma_values
    
    return result


def frama(data: ArrayLike, period: int = 9, fc_period: int = 198) -> NDArray[np.float64]:
    """
    Fractal Adaptive Moving Average (FRAMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
    fc_period : int, default 198
        Fractal cycle period
        
    Returns
    -------
    NDArray[np.float64]
        Fractal adaptive moving average values

    Raises
    ------
    ValueError
        If period or fc_period is not positive
    """
    if period <= 0 or fc_period <= 0:
        raise ValueError("Periods must be positive")
    
    data_array = np.asarray(data)
    result = np.full_like(data_array, np.nan, dtype=np.float64)
    
    result[period - 1] = np.mean(data_array[:period])
    
    for i in range(period, len(data_array)):
        if i < period * 2:
            alpha = 2 / (period + 1)
            result[i] = alpha * data_array[i] + (1 - alpha) * result[i - 1]
        else:
            n1 = period // 2
            n2 = period
            n3 = period * 2
            
            h1: float = np.max(data_array[i-n1:i])
            l1: float = np.min(data_array[i-n1:i])
            h2: float = np.max(data_array[i-n2:i-n1])
            l2: float = np.min(data_array[i-n2:i-n1])
            h3: float = np.max(data_array[i-n3:i])
            l3: float = np.min(data_array[i-n3:i])
            
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
            
            result[i] = alpha * data_array[i] + (1 - alpha) * result[i - 1]
    
    return result


def jma(data: ArrayLike, period: int = 9, phase: float = 0) -> NDArray[np.float64]:
    """
    Jurik Moving Average (JMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
    phase : float, default 0
        Phase parameter (-100 to 100)
        
    Returns
    -------
    NDArray[np.float64]
        Jurik moving average values

    Raises
    ------
    ValueError
        If period is not positive
        If phase is not between -100 and 100
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if phase < -100 or phase > 100:
        raise ValueError("Phase must be between -100 and 100")
    
    data_array = np.asarray(data)
    result = np.full_like(data_array, np.nan, dtype=np.float64)
    
    beta = 0.45 * (phase / 100) + 0.5
    
    alpha = 0.0962 / period + 0.5769
    power = np.exp(-3.067 * alpha)
    
    e0 = 0.0
    e1 = 0.0
    e2 = 0.0
    jma = data_array[0]
    result[0] = jma
    
    for i in range(1, len(data_array)):
        price_delta = data_array[i] - data_array[i-1]
        
        e0 = (1 - alpha) * e0 + alpha * price_delta
        
        e1 = (data_array[i] - jma) * power + beta * e0
        
        e2 = (1 - alpha) * e2 + alpha * e1
        
        jma += e2
        
        result[i] = jma
    
    return result


def lsma(data: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Least Squares Moving Average (LSMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Least squares moving average values

    Raises
    ------
    ValueError
        If period is not positive
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    x = np.arange(period)
    
    x_mean = np.mean(x)
    x_squared_sum: float = np.sum((x - x_mean) ** 2)
    
    for i in range(period - 1, len(data_array)):
        y = data_array[i - period + 1:i + 1]
        y_mean = np.mean(y)
        
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_squared_sum
        
        intercept = y_mean - slope * x_mean
        
        result[i] = intercept + slope * (period - 1)
    
    return result


def mcginley_dynamic(data: ArrayLike, period: int = 9, k: float = 0.6) -> NDArray[np.float64]:
    """
    McGinley Dynamic Indicator
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
    k : float, default 0.6
        Adjustment factor
        
    Returns
    -------
    NDArray[np.float64]
        McGinley dynamic indicator values

    Raises
    ------
    ValueError
        If period is not positive
        If k is not between 0 and 1
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if k < 0 or k > 1:
        raise ValueError("k must be between 0 and 1")
    
    data_array = np.asarray(data)
    result = np.full_like(data_array, np.nan, dtype=np.float64)
    
    result[period - 1] = np.mean(data_array[:period])
    
    for i in range(period, len(data_array)):
        md_prev = result[i - 1]
        price = data_array[i]
        
        result[i] = md_prev + (price - md_prev) / (k * period * np.power(price / md_prev, 4))
    
    return result


def t3(data: ArrayLike, period: int = 9, vfactor: float = 0.7) -> NDArray[np.float64]:
    """
    Tillson T3 Moving Average
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the moving average
    vfactor : float, default 0.7
        Volume factor (0-1)
        
    Returns
    -------
    NDArray[np.float64]
        T3 moving average values

    Raises
    ------
    ValueError
        If period is not positive
        If volume factor is not between 0 and 1
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if vfactor < 0 or vfactor > 1:
        raise ValueError("Volume factor must be between 0 and 1")
    
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    e1 = np.full_like(data_array, np.nan)
    
    if len(data_array) >= period:
        e1[period - 1] = np.mean(data_array[:period])
        
        for i in range(period, len(data_array)):
            e1[i] = (2 / (period + 1)) * data_array[i] + (1 - 2 / (period + 1)) * e1[i - 1]
    
    e2 = np.full_like(data_array, np.nan)
    if len(data_array) >= period * 2 - 1:
        e2[period * 2 - 2] = np.mean(e1[period-1:period*2-1])
        
        for i in range(period * 2 - 1, len(data_array)):
            e2[i] = (2 / (period + 1)) * e1[i] + (1 - 2 / (period + 1)) * e2[i - 1]
    
    e3 = np.full_like(data_array, np.nan)
    if len(data_array) >= period * 3 - 2:
        e3[period * 3 - 3] = np.mean(e2[period*2-2:period*3-2])
        
        for i in range(period * 3 - 2, len(data_array)):
            e3[i] = (2 / (period + 1)) * e2[i] + (1 - 2 / (period + 1)) * e3[i - 1]
    
    e4 = np.full_like(data_array, np.nan)
    if len(data_array) >= period * 4 - 3:
        e4[period * 4 - 4] = np.mean(e3[period*3-3:period*4-3])
        
        for i in range(period * 4 - 3, len(data_array)):
            e4[i] = (2 / (period + 1)) * e3[i] + (1 - 2 / (period + 1)) * e4[i - 1]
    
    e5 = np.full_like(data_array, np.nan)
    if len(data_array) >= period * 5 - 4:
        e5[period * 5 - 5] = np.mean(e4[period*4-4:period*5-4])
        
        for i in range(period * 5 - 4, len(data_array)):
            e5[i] = (2 / (period + 1)) * e4[i] + (1 - 2 / (period + 1)) * e5[i - 1]
    
    e6 = np.full_like(data_array, np.nan)
    if len(data_array) >= period * 6 - 5:
        e6[period * 6 - 6] = np.mean(e5[period*5-5:period*6-5])
        
        for i in range(period * 6 - 5, len(data_array)):
            e6[i] = (2 / (period + 1)) * e5[i] + (1 - 2 / (period + 1)) * e6[i - 1]
    
    c1 = -vfactor**3
    c2 = 3 * vfactor**2 + 3 * vfactor**3
    c3 = -6 * vfactor**2 - 3 * vfactor - 3 * vfactor**3
    c4 = 1 + 3 * vfactor + vfactor**3 + 3 * vfactor**2
    
    valid_indices = ~np.isnan(e6)
    if np.any(valid_indices):
        result[valid_indices] = c1 * e6[valid_indices] + c2 * e5[valid_indices] + c3 * e4[valid_indices] + c4 * e3[valid_indices]
    
    return result


def vama(data: ArrayLike, volatility: ArrayLike, period: int = 9) -> NDArray[np.float64]:
    """
    Volatility-Adjusted Moving Average (VAMA)
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    volatility : ArrayLike
        Volatility data corresponding to price data
    period : int, default 9
        Window size for the moving average
        
    Returns
    -------
    NDArray[np.float64]
        Volatility-adjusted moving average values

    Raises
    ------
    ValueError
        If period is not positive
        If price and volatility arrays have different lengths
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    
    data_array = np.asarray(data)
    volatility_array = np.asarray(volatility)
    
    if len(data_array) != len(volatility_array):
        raise ValueError("Price and volatility arrays must have the same length")
    
    result = np.full_like(data_array, np.nan, dtype=np.float64)
    
    for i in range(period - 1, len(data_array)):
        price_window = data_array[i - period + 1:i + 1]
        vol_window = volatility_array[i - period + 1:i + 1]
        
        vol_sum: float = np.sum(vol_window)
        if vol_sum == 0:
            result[i] = np.mean(price_window)
        else:
            weights = vol_window / vol_sum
            result[i] = np.sum(price_window * weights)
    
    return result


def laguerre_filter(data: ArrayLike, gamma: float = 0.8) -> NDArray[np.float64]:
    """
    Laguerre Filter
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    gamma : float, default 0.8
        Damping factor (0-1)
        
    Returns
    -------
    NDArray[np.float64]
        Laguerre filter values

    Raises
    ------
    ValueError
        If gamma is not between 0 and 1
    """
    if gamma < 0 or gamma > 1:
        raise ValueError("Gamma must be between 0 and 1")
    
    data_array = np.asarray(data)
    result = np.full_like(data_array, np.nan, dtype=np.float64)
    
    l0 = np.zeros_like(data_array)
    l1 = np.zeros_like(data_array)
    l2 = np.zeros_like(data_array)
    l3 = np.zeros_like(data_array)
    
    for i in range(1, len(data_array)):
        l0[i] = (1 - gamma) * data_array[i] + gamma * l0[i-1]
        l1[i] = -gamma * l0[i] + l0[i-1] + gamma * l1[i-1]
        l2[i] = -gamma * l1[i] + l1[i-1] + gamma * l2[i-1]
        l3[i] = -gamma * l2[i] + l2[i-1] + gamma * l3[i-1]
        
        result[i] = (l0[i] + 2*l1[i] + 2*l2[i] + l3[i]) / 6
    
    return result


def modular_filter(data: ArrayLike, period: int = 9, phase: float = 0.5) -> NDArray[np.float64]:
    """
    Modular Filter
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
    period : int, default 9
        Window size for the filter
    phase : float, default 0.5
        Phase parameter (0-1)
        
    Returns
    -------
    NDArray[np.float64]
        Modular filter values

    Raises
    ------
    ValueError
        If period is not positive
        If phase is not between 0 and 1
    """
    if period <= 0:
        raise ValueError("Period must be positive")
    if phase < 0 or phase > 1:
        raise ValueError("Phase must be between 0 and 1")
    
    data_array = np.asarray(data)
    result = np.full_like(data_array, np.nan, dtype=np.float64)
    
    alpha = 2 / (period + 1)
    
    result[0] = data_array[0]
    
    for i in range(1, len(data_array)):
        result[i] = (1 - alpha) * result[i-1] + alpha * (data_array[i] + phase * (data_array[i] - data_array[i-1]))
    
    return result


def rdma(data: ArrayLike) -> NDArray[np.float64]:
    """
    Rex Dog Moving Average (RDMA)
    
    This implementation follows the original RexDog definition, which is the average
    of six SMAs with periods 5, 9, 24, 50, 100, and 200.
    
    Parameters
    ----------
    data : ArrayLike
        Input price data
        
    Returns
    -------
    NDArray[np.float64]
        Rex Dog moving average values
    """
    data_array = np.asarray(data, dtype=np.float64)
    result = np.full_like(data_array, np.nan)
    
    periods = [5, 9, 24, 50, 100, 200]
    max_period = max(periods)
    
    sma_results = []
    for period in periods:
        weights = np.ones(period) / period
        sma_result = np.full_like(data_array, np.nan)
        convolved = np.convolve(data_array, weights, mode='valid')
        sma_result[period-1:] = convolved
        sma_results.append(sma_result)
    
    stacked_results = np.stack(sma_results, axis=0)
    
    result[max_period-1:] = np.nanmean(stacked_results[:, max_period-1:], axis=0)
    
    return result
