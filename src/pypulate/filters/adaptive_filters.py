"""
Adaptive Filters Module

This module provides implementations of various adaptive filtering algorithms
for financial time series data.
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable

def adaptive_kalman_filter(
    data: np.ndarray,
    process_variance_init: float = 1e-5,
    measurement_variance_init: float = 1e-3,
    adaptation_rate: float = 0.01,
    window_size: int = 10,
    initial_state: Optional[float] = None,
    initial_covariance: float = 1.0
) -> np.ndarray:
    """
    Apply an adaptive Kalman filter to a time series.
    
    The adaptive Kalman filter automatically adjusts its parameters based on
    the observed data, making it more robust to changing dynamics.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    process_variance_init : float, default 1e-5
        Initial process noise variance (Q)
    measurement_variance_init : float, default 1e-3
        Initial measurement noise variance (R)
    adaptation_rate : float, default 0.01
        Rate at which the filter adapts to changes
    window_size : int, default 10
        Size of the window for innovation estimation
    initial_state : float, optional
        Initial state estimate. If None, the first data point is used
    initial_covariance : float, default 1.0
        Initial estimate covariance
        
    Returns
    -------
    np.ndarray
        Filtered time series
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import adaptive_kalman_filter
    >>> # Create noisy data with changing dynamics
    >>> x = np.linspace(0, 10, 200)
    >>> true_signal = np.sin(x) + 0.1 * x
    >>> noise_level = 0.1 * (1 + np.sin(x/2))  # Changing noise level
    >>> noisy_signal = true_signal + noise_level * np.random.randn(len(x))
    >>> # Apply adaptive Kalman filter
    >>> filtered_signal = adaptive_kalman_filter(noisy_signal, adaptation_rate=0.05)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Initialize state and filtered data
    filtered_data = np.zeros(n)
    
    # Initialize state estimate and covariance
    if initial_state is None:
        state_estimate = data[0]
    else:
        state_estimate = initial_state
    
    estimate_covariance = initial_covariance
    
    # Kalman filter parameters
    Q = process_variance_init  # Process noise variance
    R = measurement_variance_init  # Measurement noise variance
    
    # Innovation history for adaptive estimation
    innovations = np.zeros(window_size)
    innovation_idx = 0
    
    # Apply adaptive Kalman filter
    for i in range(n):
        # Prediction step
        # For a simple random walk model, the prediction is just the previous state
        predicted_state = state_estimate
        predicted_covariance = estimate_covariance + Q
        
        # Update step
        kalman_gain = predicted_covariance / (predicted_covariance + R)
        innovation = data[i] - predicted_state
        state_estimate = predicted_state + kalman_gain * innovation
        estimate_covariance = (1 - kalman_gain) * predicted_covariance
        
        # Store filtered value
        filtered_data[i] = state_estimate
        
        # Update innovation history
        innovations[innovation_idx] = innovation
        innovation_idx = (innovation_idx + 1) % window_size
        
        # Adapt measurement noise variance based on innovation statistics
        if i >= window_size:
            innovation_variance = np.var(innovations)
            R = (1 - adaptation_rate) * R + adaptation_rate * innovation_variance
        
        # Adapt process noise variance based on prediction error
        if i > 0:
            prediction_error = state_estimate - filtered_data[i-1]
            Q = (1 - adaptation_rate) * Q + adaptation_rate * prediction_error**2
    
    return filtered_data

def least_mean_squares_filter(
    data: np.ndarray,
    desired: Optional[np.ndarray] = None,
    filter_length: int = 5,
    mu: float = 0.01,
    initial_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a Least Mean Squares (LMS) adaptive filter to a time series.
    
    The LMS algorithm is an adaptive filter that adjusts its coefficients to
    minimize the mean square error between the desired signal and the filter output.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    desired : np.ndarray, optional
        Desired signal. If None, a delayed version of the input is used
    filter_length : int, default 5
        Length of the adaptive filter
    mu : float, default 0.01
        Step size (learning rate) of the adaptation
    initial_weights : np.ndarray, optional
        Initial filter weights. If None, zeros are used
        
    Returns
    -------
    tuple of np.ndarray
        Tuple containing (filtered_data, filter_weights)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import least_mean_squares_filter
    >>> # Create noisy data
    >>> x = np.linspace(0, 10, 1000)
    >>> clean_signal = np.sin(2 * np.pi * 0.05 * x)
    >>> noise = 0.2 * np.random.randn(len(x))
    >>> noisy_signal = clean_signal + noise
    >>> # Apply LMS filter
    >>> filtered_signal, weights = least_mean_squares_filter(noisy_signal, filter_length=10, mu=0.02)
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # If desired signal is not provided, use a delayed version of the input
    if desired is None:
        delay = filter_length // 2
        desired = np.zeros_like(data)
        desired[delay:] = data[:-delay] if delay > 0 else data
    
    # Initialize filter weights
    if initial_weights is None:
        weights = np.zeros(filter_length)
    else:
        weights = initial_weights.copy()
    
    # Initialize output signal
    filtered_data = np.zeros(n)
    
    # Apply LMS filter
    for i in range(filter_length - 1, n):
        # Get input vector (window of the signal)
        x = data[i - filter_length + 1:i + 1]
        
        # Calculate filter output
        y = np.dot(weights, x)
        
        # Calculate error
        e = desired[i] - y
        
        # Update weights
        weights = weights + mu * e * x
        
        # Store filtered value
        filtered_data[i] = y
    
    # Fill initial values
    filtered_data[:filter_length - 1] = data[:filter_length - 1]
    
    return filtered_data, weights

def recursive_least_squares_filter(
    data: np.ndarray,
    desired: Optional[np.ndarray] = None,
    filter_length: int = 5,
    forgetting_factor: float = 0.99,
    delta: float = 1.0,
    initial_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a Recursive Least Squares (RLS) adaptive filter to a time series.
    
    The RLS algorithm is an adaptive filter that recursively finds the filter
    coefficients that minimize a weighted linear least squares cost function
    related to the input signals.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    desired : np.ndarray, optional
        Desired signal. If None, a delayed version of the input is used
    filter_length : int, default 5
        Length of the adaptive filter
    forgetting_factor : float, default 0.99
        Forgetting factor (0 < lambda <= 1)
    delta : float, default 1.0
        Regularization parameter for the initial correlation matrix
    initial_weights : np.ndarray, optional
        Initial filter weights. If None, zeros are used
        
    Returns
    -------
    tuple of np.ndarray
        Tuple containing (filtered_data, filter_weights)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import recursive_least_squares_filter
    >>> # Create noisy data
    >>> x = np.linspace(0, 10, 1000)
    >>> clean_signal = np.sin(2 * np.pi * 0.05 * x)
    >>> noise = 0.2 * np.random.randn(len(x))
    >>> noisy_signal = clean_signal + noise
    >>> # Apply RLS filter
    >>> filtered_signal, weights = recursive_least_squares_filter(
    ...     noisy_signal, filter_length=10, forgetting_factor=0.99
    ... )
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # If desired signal is not provided, use a delayed version of the input
    if desired is None:
        delay = filter_length // 2
        desired = np.zeros_like(data)
        desired[delay:] = data[:-delay] if delay > 0 else data
    
    # Initialize filter weights
    if initial_weights is None:
        weights = np.zeros(filter_length)
    else:
        weights = initial_weights.copy()
    
    # Initialize correlation matrix
    P = np.eye(filter_length) / delta
    
    # Initialize output signal
    filtered_data = np.zeros(n)
    
    # Apply RLS filter
    for i in range(filter_length - 1, n):
        # Get input vector (window of the signal)
        x = data[i - filter_length + 1:i + 1]
        
        # Calculate filter output
        y = np.dot(weights, x)
        
        # Calculate error
        e = desired[i] - y
        
        # Calculate gain vector
        k = P @ x / (forgetting_factor + x @ P @ x)
        
        # Update weights
        weights = weights + k * e
        
        # Update correlation matrix
        P = (P - np.outer(k, x) @ P) / forgetting_factor
        
        # Store filtered value
        filtered_data[i] = y
    
    # Fill initial values
    filtered_data[:filter_length - 1] = data[:filter_length - 1]
    
    return filtered_data, weights 