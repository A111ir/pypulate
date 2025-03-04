"""
Kalman Filters Module

This module provides implementations of various Kalman filter algorithms
for financial time series data.
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable

def kalman_filter(
    data: np.ndarray,
    process_variance: float = 1e-5,
    measurement_variance: float = 1e-3,
    initial_state: Optional[float] = None,
    initial_covariance: float = 1.0
) -> np.ndarray:
    """
    Apply a standard Kalman filter to a time series.
    
    The Kalman filter is an optimal estimator that infers parameters of interest
    from indirect, inaccurate and uncertain observations. It's recursive so new
    measurements can be processed as they arrive.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data
    process_variance : float, default 1e-5
        Process noise variance (Q)
    measurement_variance : float, default 1e-3
        Measurement noise variance (R)
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
    >>> from pypulate.filters import kalman_filter
    >>> # Create noisy data
    >>> x = np.linspace(0, 10, 100)
    >>> true_signal = np.sin(x)
    >>> noisy_signal = true_signal + np.random.normal(0, 0.1, len(x))
    >>> # Apply Kalman filter
    >>> filtered_signal = kalman_filter(noisy_signal)
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
    Q = process_variance  # Process noise variance
    R = measurement_variance  # Measurement noise variance
    
    # Apply Kalman filter
    for i in range(n):
        # Prediction step
        # For a simple random walk model, the prediction is just the previous state
        predicted_state = state_estimate
        predicted_covariance = estimate_covariance + Q
        
        # Update step
        kalman_gain = predicted_covariance / (predicted_covariance + R)
        state_estimate = predicted_state + kalman_gain * (data[i] - predicted_state)
        estimate_covariance = (1 - kalman_gain) * predicted_covariance
        
        # Store filtered value
        filtered_data[i] = state_estimate
    
    return filtered_data

def extended_kalman_filter(
    data: np.ndarray,
    state_transition_func: Callable,
    observation_func: Callable,
    process_jacobian_func: Callable,
    observation_jacobian_func: Callable,
    process_covariance: np.ndarray,
    observation_covariance: np.ndarray,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply an Extended Kalman Filter (EKF) to a time series with non-linear dynamics.
    
    The EKF is a nonlinear version of the Kalman filter that linearizes about the
    current mean and covariance. It's used when the state transition or observation
    models are non-linear.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data (observations)
    state_transition_func : callable
        Function that computes the state transition (f)
    observation_func : callable
        Function that computes the observation from state (h)
    process_jacobian_func : callable
        Function that computes the Jacobian of the state transition function
    observation_jacobian_func : callable
        Function that computes the Jacobian of the observation function
    process_covariance : np.ndarray
        Process noise covariance matrix (Q)
    observation_covariance : np.ndarray
        Observation noise covariance matrix (R)
    initial_state : np.ndarray, optional
        Initial state estimate. If None, zeros are used
    initial_covariance : np.ndarray, optional
        Initial estimate covariance matrix. If None, identity is used
        
    Returns
    -------
    np.ndarray
        Filtered time series (state estimates)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import extended_kalman_filter
    >>> # Define non-linear system
    >>> def state_transition(x):
    ...     # Non-linear state transition function
    ...     return np.array([x[0] + x[1], 0.5 * x[1]])
    >>> def observation(x):
    ...     # Non-linear observation function
    ...     return np.array([np.sin(x[0])])
    >>> def process_jacobian(x):
    ...     # Jacobian of state transition function
    ...     return np.array([[1, 1], [0, 0.5]])
    >>> def observation_jacobian(x):
    ...     # Jacobian of observation function
    ...     return np.array([[np.cos(x[0]), 0]])
    >>> # Create data
    >>> n = 100
    >>> true_states = np.zeros((n, 2))
    >>> true_states[0] = [0, 1]
    >>> for i in range(1, n):
    ...     true_states[i] = state_transition(true_states[i-1])
    >>> observations = np.array([observation(x)[0] for x in true_states])
    >>> observations += np.random.normal(0, 0.1, n)
    >>> # Apply EKF
    >>> Q = np.eye(2) * 0.01  # Process noise covariance
    >>> R = np.array([[0.1]])  # Observation noise covariance
    >>> filtered_states = extended_kalman_filter(
    ...     observations, state_transition, observation,
    ...     process_jacobian, observation_jacobian, Q, R
    ... )
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Determine state dimension from process covariance
    state_dim = process_covariance.shape[0]
    
    # Initialize state and filtered data
    if initial_state is None:
        state_estimate = np.zeros(state_dim)
    else:
        state_estimate = initial_state
    
    if initial_covariance is None:
        estimate_covariance = np.eye(state_dim)
    else:
        estimate_covariance = initial_covariance
    
    # Prepare output array
    filtered_states = np.zeros((n, state_dim))
    
    # Apply Extended Kalman Filter
    for i in range(n):
        # Prediction step
        predicted_state = state_transition_func(state_estimate)
        F = process_jacobian_func(state_estimate)  # Jacobian of state transition
        predicted_covariance = F @ estimate_covariance @ F.T + process_covariance
        
        # Update step
        predicted_measurement = observation_func(predicted_state)
        H = observation_jacobian_func(predicted_state)  # Jacobian of observation
        
        innovation = data[i] - predicted_measurement
        innovation_covariance = H @ predicted_covariance @ H.T + observation_covariance
        
        kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(innovation_covariance)
        
        state_estimate = predicted_state + kalman_gain @ innovation
        estimate_covariance = (np.eye(state_dim) - kalman_gain @ H) @ predicted_covariance
        
        # Store filtered state
        filtered_states[i] = state_estimate
    
    return filtered_states

def unscented_kalman_filter(
    data: np.ndarray,
    state_transition_func: Callable,
    observation_func: Callable,
    process_covariance: np.ndarray,
    observation_covariance: np.ndarray,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0
) -> np.ndarray:
    """
    Apply an Unscented Kalman Filter (UKF) to a time series with non-linear dynamics.
    
    The UKF uses the unscented transform to pick a minimal set of sample points (sigma points)
    around the mean. These sigma points are then propagated through the non-linear functions,
    and the mean and covariance of the estimate are recovered.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data (observations)
    state_transition_func : callable
        Function that computes the state transition
    observation_func : callable
        Function that computes the observation from state
    process_covariance : np.ndarray
        Process noise covariance matrix (Q)
    observation_covariance : np.ndarray
        Observation noise covariance matrix (R)
    initial_state : np.ndarray, optional
        Initial state estimate. If None, zeros are used
    initial_covariance : np.ndarray, optional
        Initial estimate covariance matrix. If None, identity is used
    alpha : float, default 1e-3
        Spread of sigma points around mean
    beta : float, default 2.0
        Prior knowledge about distribution (2 is optimal for Gaussian)
    kappa : float, default 0.0
        Secondary scaling parameter
        
    Returns
    -------
    np.ndarray
        Filtered time series (state estimates)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import unscented_kalman_filter
    >>> # Define non-linear system
    >>> def state_transition(x):
    ...     # Non-linear state transition function
    ...     return np.array([x[0] + x[1], 0.5 * x[1]])
    >>> def observation(x):
    ...     # Non-linear observation function
    ...     return np.array([np.sin(x[0])])
    >>> # Create data
    >>> n = 100
    >>> true_states = np.zeros((n, 2))
    >>> true_states[0] = [0, 1]
    >>> for i in range(1, n):
    ...     true_states[i] = state_transition(true_states[i-1])
    >>> observations = np.array([observation(x)[0] for x in true_states])
    >>> observations += np.random.normal(0, 0.1, n)
    >>> # Apply UKF
    >>> Q = np.eye(2) * 0.01  # Process noise covariance
    >>> R = np.array([[0.1]])  # Observation noise covariance
    >>> filtered_states = unscented_kalman_filter(
    ...     observations, state_transition, observation, Q, R
    ... )
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Determine state dimension from process covariance
    state_dim = process_covariance.shape[0]
    
    # Initialize state and filtered data
    if initial_state is None:
        state_estimate = np.zeros(state_dim)
    else:
        state_estimate = initial_state
    
    if initial_covariance is None:
        estimate_covariance = np.eye(state_dim)
    else:
        estimate_covariance = initial_covariance
    
    # Prepare output array
    filtered_states = np.zeros((n, state_dim))
    
    # UKF parameters
    lambda_param = alpha**2 * (state_dim + kappa) - state_dim
    n_sigma_points = 2 * state_dim + 1
    
    # Weights for mean and covariance
    weights_mean = np.zeros(n_sigma_points)
    weights_cov = np.zeros(n_sigma_points)
    
    weights_mean[0] = lambda_param / (state_dim + lambda_param)
    weights_cov[0] = weights_mean[0] + (1 - alpha**2 + beta)
    
    for i in range(1, n_sigma_points):
        weights_mean[i] = 1 / (2 * (state_dim + lambda_param))
        weights_cov[i] = weights_mean[i]
    
    # Apply Unscented Kalman Filter
    for i in range(n):
        # Generate sigma points
        sigma_points = np.zeros((n_sigma_points, state_dim))
        sigma_points[0] = state_estimate
        
        # Calculate square root of covariance matrix
        L = np.linalg.cholesky((state_dim + lambda_param) * estimate_covariance)
        
        for j in range(state_dim):
            sigma_points[j+1] = state_estimate + L[j]
            sigma_points[j+1+state_dim] = state_estimate - L[j]
        
        # Prediction step
        # Propagate sigma points through state transition function
        predicted_sigma_points = np.array([state_transition_func(s) for s in sigma_points])
        
        # Calculate predicted mean and covariance
        predicted_state = np.zeros(state_dim)
        for j in range(n_sigma_points):
            predicted_state += weights_mean[j] * predicted_sigma_points[j]
        
        predicted_covariance = process_covariance.copy()
        for j in range(n_sigma_points):
            diff = predicted_sigma_points[j] - predicted_state
            predicted_covariance += weights_cov[j] * np.outer(diff, diff)
        
        # Propagate sigma points through observation function
        predicted_measurements = np.array([observation_func(s) for s in predicted_sigma_points])
        
        # Calculate predicted measurement mean and covariance
        measurement_dim = predicted_measurements[0].shape[0]
        predicted_measurement = np.zeros(measurement_dim)
        for j in range(n_sigma_points):
            predicted_measurement += weights_mean[j] * predicted_measurements[j]
        
        measurement_covariance = observation_covariance.copy()
        cross_correlation = np.zeros((state_dim, measurement_dim))
        
        for j in range(n_sigma_points):
            diff_state = predicted_sigma_points[j] - predicted_state
            diff_meas = predicted_measurements[j] - predicted_measurement
            
            measurement_covariance += weights_cov[j] * np.outer(diff_meas, diff_meas)
            cross_correlation += weights_cov[j] * np.outer(diff_state, diff_meas)
        
        # Update step
        kalman_gain = cross_correlation @ np.linalg.inv(measurement_covariance)
        
        innovation = data[i] - predicted_measurement
        state_estimate = predicted_state + kalman_gain @ innovation
        estimate_covariance = predicted_covariance - kalman_gain @ measurement_covariance @ kalman_gain.T
        
        # Store filtered state
        filtered_states[i] = state_estimate
    
    return filtered_states 