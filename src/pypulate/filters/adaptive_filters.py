"""
Adaptive Filters Module

This module provides implementations of various adaptive filtering algorithms
for financial time series data.
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable, List, cast
from numpy.typing import NDArray, ArrayLike

def adaptive_kalman_filter(
    data: ArrayLike,
    process_variance_init: float = 1e-5,
    measurement_variance_init: float = 1e-3,
    adaptation_rate: float = 0.01,
    window_size: int = 10,
    initial_state: Optional[float] = None,
    initial_covariance: float = 1.0
) -> NDArray[np.float64]:
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
        
    """
    data = np.asarray(data)
    n = len(data)
    
    filtered_data: NDArray[np.float64] = np.zeros(n)
    
    state_estimate: float
    if initial_state is None:
        state_estimate = float(data[0])
    else:
        state_estimate = initial_state
    
    estimate_covariance: float = initial_covariance
    
    Q: float = process_variance_init 
    R: float = measurement_variance_init 
    
    innovations: NDArray[np.float64] = np.zeros(window_size)
    innovation_idx: int = 0
    
    for i in range(n):
        predicted_state: float = state_estimate
        predicted_covariance: float = estimate_covariance + Q
        
        kalman_gain: float = predicted_covariance / (predicted_covariance + R)
        innovation: float = float(data[i]) - predicted_state
        state_estimate = predicted_state + kalman_gain * innovation
        estimate_covariance = (1 - kalman_gain) * predicted_covariance
        
        filtered_data[i] = state_estimate
        
        innovations[innovation_idx] = innovation
        innovation_idx = (innovation_idx + 1) % window_size
        
        if i >= window_size:
            innovation_variance: float = float(np.var(innovations))
            R = (1 - adaptation_rate) * R + adaptation_rate * innovation_variance
        
        if i > 0:
            prediction_error: float = state_estimate - filtered_data[i-1]
            Q = (1 - adaptation_rate) * Q + adaptation_rate * prediction_error**2
    
    return filtered_data

def least_mean_squares_filter(
    data: np.ndarray,
    desired: Optional[np.ndarray] = None,
    filter_length: int = 5,
    mu: float = 0.01,
    initial_weights: Optional[np.ndarray] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        
    """
    data = np.asarray(data)
    n = len(data)
    
    if desired is None:
        delay = filter_length // 2
        desired = np.zeros_like(data)
        desired[delay:] = data[:-delay] if delay > 0 else data
    
    filter_weights: NDArray[np.float64]
    if initial_weights is None:
        filter_weights = np.zeros(filter_length)
    else:
        filter_weights = initial_weights.copy()
    
    filtered_data: NDArray[np.float64] = np.zeros(n)
    
    for i in range(filter_length - 1, n):
        x = data[i - filter_length + 1:i + 1]
        
        y: float = float(np.dot(filter_weights, x))
        
        e: float = float(desired[i]) - y
        
        filter_weights = filter_weights + mu * e * x
        
        filtered_data[i] = y
    
    filtered_data[:filter_length - 1] = data[:filter_length - 1]
    
    return filtered_data, filter_weights

def recursive_least_squares_filter(
    data: ArrayLike,
    desired: Optional[np.ndarray] = None,
    filter_length: int = 5,
    forgetting_factor: float = 0.99,
    delta: float = 1.0,
    initial_weights: Optional[np.ndarray] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        
    """
    data = np.asarray(data)
    n = len(data)
    
    if desired is None:
        delay = filter_length // 2
        desired = np.zeros_like(data)
        desired[delay:] = data[:-delay] if delay > 0 else data
    
    filter_weights: NDArray[np.float64]
    if initial_weights is None:
        filter_weights = np.zeros(filter_length)
    else:
        filter_weights = initial_weights.copy()
    
    P: NDArray[np.float64] = np.eye(filter_length) / delta
    
    filtered_data: NDArray[np.float64] = np.zeros(n)
    
    for i in range(filter_length - 1, n):
        x = data[i - filter_length + 1:i + 1]
        
        y: float = float(np.dot(filter_weights, x))
        
        e: float = float(desired[i]) - y
        
        k: NDArray[np.float64] = P @ x / (forgetting_factor + float(x @ P @ x))
        
        filter_weights = filter_weights + k * e
        
        P = (P - np.outer(k, x) @ P) / forgetting_factor
        
        filtered_data[i] = y
    
    filtered_data[:filter_length - 1] = data[:filter_length - 1]
    
    return filtered_data, filter_weights 