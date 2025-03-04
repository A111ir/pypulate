"""
Signal Filters Module

This module provides implementations of various signal processing filters
for financial time series data.
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable
from scipy import signal

def butterworth_filter(
    data: np.ndarray,
    cutoff: Union[float, Tuple[float, float]],
    order: int = 4,
    filter_type: str = 'lowpass',
    fs: float = 1.0
) -> np.ndarray:
    """
    Apply a Butterworth filter to a time series.
    
    The Butterworth filter is a type of signal processing filter designed to have
    a frequency response as flat as possible in the passband.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    cutoff : float or tuple of float
        Cutoff frequency. For lowpass and highpass, this is a scalar.
        For bandpass and bandstop, this is a tuple of (low, high)
    order : int, default 4
        Filter order
    filter_type : str, default 'lowpass'
        Filter type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
    fs : float, default 1.0
        Sampling frequency
        
    Returns
    -------
    np.ndarray
        Filtered time series
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import butterworth_filter
    >>> # Create noisy data
    >>> x = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2 * np.pi * 0.05 * x) + 0.5 * np.sin(2 * np.pi * 0.25 * x)
    >>> # Apply lowpass filter to remove high frequency component
    >>> filtered = butterworth_filter(signal, cutoff=0.1, filter_type='lowpass', fs=1.0)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # Normalize cutoff frequency
    nyquist = 0.5 * fs
    if isinstance(cutoff, tuple):
        cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
    else:
        cutoff = cutoff / nyquist
    
    # Design filter
    b, a = signal.butter(order, cutoff, btype=filter_type)
    
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def chebyshev_filter(
    data: np.ndarray,
    cutoff: Union[float, Tuple[float, float]],
    order: int = 4,
    ripple: float = 1.0,
    filter_type: str = 'lowpass',
    fs: float = 1.0,
    type_num: int = 1
) -> np.ndarray:
    """
    Apply a Chebyshev filter to a time series.
    
    The Chebyshev filter is a filter with steeper roll-off than the Butterworth
    but more passband ripple (type I) or stopband ripple (type II).
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    cutoff : float or tuple of float
        Cutoff frequency. For lowpass and highpass, this is a scalar.
        For bandpass and bandstop, this is a tuple of (low, high)
    order : int, default 4
        Filter order
    ripple : float, default 1.0
        Maximum ripple allowed in the passband (type I) or stopband (type II)
    filter_type : str, default 'lowpass'
        Filter type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
    fs : float, default 1.0
        Sampling frequency
    type_num : int, default 1
        Type of Chebyshev filter: 1 for Type I, 2 for Type II
        
    Returns
    -------
    np.ndarray
        Filtered time series
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import chebyshev_filter
    >>> # Create noisy data
    >>> x = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2 * np.pi * 0.05 * x) + 0.5 * np.sin(2 * np.pi * 0.25 * x)
    >>> # Apply Chebyshev type I lowpass filter
    >>> filtered = chebyshev_filter(signal, cutoff=0.1, ripple=0.5, type_num=1)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # Normalize cutoff frequency
    nyquist = 0.5 * fs
    if isinstance(cutoff, tuple):
        cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
    else:
        cutoff = cutoff / nyquist
    
    # Design filter
    if type_num == 1:
        b, a = signal.cheby1(order, ripple, cutoff, btype=filter_type)
    else:
        b, a = signal.cheby2(order, ripple, cutoff, btype=filter_type)
    
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def savitzky_golay_filter(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
    delta: float = 1.0
) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter to a time series.
    
    The Savitzky-Golay filter is a digital filter that can be applied to a set of
    digital data points to smooth the data by increasing the signal-to-noise ratio
    without greatly distorting the signal.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    window_length : int, default 11
        Length of the filter window (must be odd)
    polyorder : int, default 3
        Order of the polynomial used to fit the samples (must be less than window_length)
    deriv : int, default 0
        Order of the derivative to compute
    delta : float, default 1.0
        Spacing of the samples to which the filter is applied
        
    Returns
    -------
    np.ndarray
        Filtered time series
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import savitzky_golay_filter
    >>> # Create noisy data
    >>> x = np.linspace(0, 10, 100)
    >>> signal = np.sin(x) + np.random.normal(0, 0.1, len(x))
    >>> # Apply Savitzky-Golay filter
    >>> filtered = savitzky_golay_filter(signal, window_length=11, polyorder=3)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Apply filter
    filtered_data = signal.savgol_filter(data, window_length, polyorder, deriv=deriv, delta=delta)
    
    return filtered_data

def wiener_filter(
    data: np.ndarray,
    mysize: Union[int, Tuple[int, ...]] = 3,
    noise: Optional[float] = None
) -> np.ndarray:
    """
    Apply a Wiener filter to a time series.
    
    The Wiener filter is a filter used to produce an estimate of a desired or target
    signal by linear filtering of an observed noisy signal.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    mysize : int or tuple of int, default 3
        Size of the filter window
    noise : float, optional
        Estimate of the noise power. If None, it's estimated from the data
        
    Returns
    -------
    np.ndarray
        Filtered time series
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import wiener_filter
    >>> # Create noisy data
    >>> x = np.linspace(0, 10, 100)
    >>> signal = np.sin(x) + np.random.normal(0, 0.1, len(x))
    >>> # Apply Wiener filter
    >>> filtered = wiener_filter(signal, mysize=5)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # Apply filter
    filtered_data = signal.wiener(data, mysize=mysize, noise=noise)
    
    return filtered_data

def median_filter(
    data: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply a median filter to a time series.
    
    The median filter is a nonlinear digital filtering technique used to remove noise
    from a signal. It replaces each entry with the median of neighboring entries.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    kernel_size : int, default 3
        Size of the filter kernel
        
    Returns
    -------
    np.ndarray
        Filtered time series
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import median_filter
    >>> # Create noisy data with outliers
    >>> x = np.linspace(0, 10, 100)
    >>> signal = np.sin(x)
    >>> signal[10] = 5  # Add outlier
    >>> signal[50] = -5  # Add outlier
    >>> # Apply median filter to remove outliers
    >>> filtered = median_filter(signal, kernel_size=5)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # Apply filter
    filtered_data = signal.medfilt(data, kernel_size=kernel_size)
    
    return filtered_data

def hampel_filter(
    data: np.ndarray,
    window_size: int = 5,
    n_sigmas: float = 3.0
) -> np.ndarray:
    """
    Apply a Hampel filter to a time series.
    
    The Hampel filter is used to identify and replace outliers in a time series.
    It uses the median and the median absolute deviation (MAD) to identify outliers.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    window_size : int, default 5
        Size of the window (number of points on each side of the current point)
    n_sigmas : float, default 3.0
        Number of standard deviations to use for outlier detection
        
    Returns
    -------
    np.ndarray
        Filtered time series
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import hampel_filter
    >>> # Create noisy data with outliers
    >>> x = np.linspace(0, 10, 100)
    >>> signal = np.sin(x)
    >>> signal[10] = 5  # Add outlier
    >>> signal[50] = -5  # Add outlier
    >>> # Apply Hampel filter to remove outliers
    >>> filtered = hampel_filter(signal, window_size=5, n_sigmas=3.0)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Create output array
    filtered_data = data.copy()
    
    # Apply Hampel filter
    for i in range(n):
        # Define window indices
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        
        # Get window values
        window = data[start:end]
        
        # Calculate median and MAD
        median_val = np.median(window)
        mad = np.median(np.abs(window - median_val))
        
        # Scale MAD (assuming normal distribution)
        sigma = 1.4826 * mad
        
        # Check if the point is an outlier
        if np.abs(data[i] - median_val) > n_sigmas * sigma:
            filtered_data[i] = median_val
    
    return filtered_data

def hodrick_prescott_filter(
    data: np.ndarray,
    lambda_param: float = 1600.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Hodrick-Prescott filter to decompose a time series into trend and cycle components.
    
    The Hodrick-Prescott filter is a mathematical tool used in macroeconomics to separate
    the cyclical component of a time series from raw data.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    lambda_param : float, default 1600.0
        Smoothing parameter. The larger the value, the smoother the trend component
        
    Returns
    -------
    tuple of np.ndarray
        Tuple containing (trend, cycle) components
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import hodrick_prescott_filter
    >>> # Create data with trend and cycle
    >>> x = np.linspace(0, 10, 100)
    >>> trend = 0.1 * x**2
    >>> cycle = np.sin(2 * np.pi * 0.1 * x)
    >>> data = trend + cycle
    >>> # Apply Hodrick-Prescott filter
    >>> trend_component, cycle_component = hodrick_prescott_filter(data, lambda_param=100)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Create the second difference matrix
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    
    # Calculate trend component
    I = np.eye(n)
    trend = np.linalg.solve(I + lambda_param * D.T @ D, data)
    
    # Calculate cycle component
    cycle = data - trend
    
    return trend, cycle

def baxter_king_filter(
    data: np.ndarray,
    low: float = 6,
    high: float = 32,
    K: int = 12
) -> np.ndarray:
    """
    Apply the Baxter-King bandpass filter to extract business cycle components.
    
    The Baxter-King filter is a bandpass filter used to extract business cycle
    components from macroeconomic time series.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    low : float, default 6
        Minimum period of oscillations (in number of observations)
    high : float, default 32
        Maximum period of oscillations (in number of observations)
    K : int, default 12
        Number of terms in the approximation (lead/lag)
        
    Returns
    -------
    np.ndarray
        Filtered time series (cycle component)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import baxter_king_filter
    >>> # Create data with trend and multiple cycles
    >>> x = np.linspace(0, 100, 1000)
    >>> trend = 0.01 * x
    >>> long_cycle = np.sin(2 * np.pi * x / 100)  # Period of 100
    >>> business_cycle = np.sin(2 * np.pi * x / 20)  # Period of 20
    >>> short_cycle = np.sin(2 * np.pi * x / 5)  # Period of 5
    >>> data = trend + long_cycle + business_cycle + short_cycle
    >>> # Extract business cycle component (periods between 8 and 32)
    >>> cycle = baxter_king_filter(data, low=8, high=32, K=12)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Convert periods to frequencies
    low_freq = 2 * np.pi / high
    high_freq = 2 * np.pi / low
    
    # Create filter weights
    a = np.zeros(2*K+1)
    a[K] = (high_freq - low_freq) / np.pi
    for i in range(1, K+1):
        a[K+i] = (np.sin(high_freq * i) - np.sin(low_freq * i)) / (np.pi * i)
        a[K-i] = a[K+i]
    
    # Make sure weights sum to zero (remove zero frequency)
    a -= np.mean(a)
    
    # Apply filter
    filtered_data = np.zeros(n)
    for i in range(K, n-K):
        filtered_data[i] = np.sum(a * data[i-K:i+K+1])
    
    # Set NaN for the K first and last observations
    filtered_data[:K] = np.nan
    filtered_data[n-K:] = np.nan
    
    return filtered_data 