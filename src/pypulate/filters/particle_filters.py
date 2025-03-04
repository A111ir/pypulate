"""
Particle Filters Module

This module provides implementations of particle filtering algorithms
for financial time series data.
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable

def particle_filter(
    data: np.ndarray,
    state_transition_func: Callable,
    observation_func: Callable,
    process_noise_func: Callable,
    observation_likelihood_func: Callable,
    n_particles: int = 100,
    initial_state_func: Optional[Callable] = None,
    resample_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a particle filter to a time series.
    
    The particle filter is a sequential Monte Carlo method that uses a set of particles
    (samples) to represent the posterior distribution of some stochastic process
    given noisy and/or partial observations.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data (observations)
    state_transition_func : callable
        Function that propagates particles through the state transition model
    observation_func : callable
        Function that computes the expected observation from a state
    process_noise_func : callable
        Function that adds process noise to particles
    observation_likelihood_func : callable
        Function that computes the likelihood of an observation given a state
    n_particles : int, default 100
        Number of particles
    initial_state_func : callable, optional
        Function that generates initial particles. If None, a default is used
    resample_threshold : float, default 0.5
        Threshold for effective sample size ratio below which resampling occurs
        
    Returns
    -------
    tuple of np.ndarray
        Tuple containing (filtered_states, particle_weights)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import particle_filter
    >>> # Define model functions
    >>> def state_transition(particles):
    ...     # Simple random walk model
    ...     return particles
    >>> def process_noise(particles):
    ...     # Add Gaussian noise
    ...     return particles + np.random.normal(0, 0.1, particles.shape)
    >>> def observation_func(state):
    ...     # Identity observation model
    ...     return state
    >>> def observation_likelihood(observation, predicted_observation):
    ...     # Gaussian likelihood
    ...     return np.exp(-0.5 * ((observation - predicted_observation) / 0.1) ** 2)
    >>> def initial_state(n):
    ...     # Initial particles from normal distribution
    ...     return np.random.normal(0, 1, n)
    >>> # Create data
    >>> true_states = np.cumsum(np.random.normal(0, 0.1, 100))
    >>> observations = true_states + np.random.normal(0, 0.1, 100)
    >>> # Apply particle filter
    >>> filtered_states, weights = particle_filter(
    ...     observations, state_transition, observation_func,
    ...     process_noise, observation_likelihood, n_particles=1000,
    ...     initial_state_func=initial_state
    ... )
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Initialize particles
    if initial_state_func is None:
        # Default initialization: normal distribution around first observation
        particles = np.random.normal(data[0], 1.0, n_particles)
    else:
        particles = initial_state_func(n_particles)
    
    # Initialize weights
    weights = np.ones(n_particles) / n_particles
    
    # Initialize output arrays
    filtered_states = np.zeros(n)
    all_weights = np.zeros((n, n_particles))
    
    # Apply particle filter
    for i in range(n):
        # Propagate particles through state transition model
        particles = state_transition_func(particles)
        
        # Add process noise
        particles = process_noise_func(particles)
        
        # Calculate predicted observations
        predicted_observations = observation_func(particles)
        
        # Update weights based on observation likelihood
        likelihood = observation_likelihood_func(data[i], predicted_observations)
        weights = weights * likelihood
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all weights are zero, reset to uniform
            weights = np.ones(n_particles) / n_particles
        
        # Store weights
        all_weights[i] = weights
        
        # Calculate effective sample size
        n_eff = 1.0 / np.sum(weights ** 2)
        
        # Resample if effective sample size is too low
        if n_eff / n_particles < resample_threshold:
            indices = np.random.choice(n_particles, size=n_particles, p=weights)
            particles = particles[indices]
            weights = np.ones(n_particles) / n_particles
        
        # Calculate filtered state (weighted average of particles)
        filtered_states[i] = np.sum(particles * weights)
    
    return filtered_states, all_weights

def bootstrap_particle_filter(
    data: np.ndarray,
    state_transition_func: Callable,
    observation_func: Callable,
    process_noise_std: float = 0.1,
    observation_noise_std: float = 0.1,
    n_particles: int = 100,
    initial_state_mean: Optional[float] = None,
    initial_state_std: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a bootstrap particle filter to a time series.
    
    The bootstrap particle filter is a simplified version of the particle filter
    that resamples at every step and uses the state transition prior as the proposal.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series data (observations)
    state_transition_func : callable
        Function that propagates particles through the state transition model
    observation_func : callable
        Function that computes the expected observation from a state
    process_noise_std : float, default 0.1
        Standard deviation of the process noise
    observation_noise_std : float, default 0.1
        Standard deviation of the observation noise
    n_particles : int, default 100
        Number of particles
    initial_state_mean : float, optional
        Mean of the initial state distribution. If None, the first observation is used
    initial_state_std : float, default 1.0
        Standard deviation of the initial state distribution
        
    Returns
    -------
    tuple of np.ndarray
        Tuple containing (filtered_states, particle_weights)
        
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.filters import bootstrap_particle_filter
    >>> # Define model functions
    >>> def state_transition(particles):
    ...     # Simple random walk model
    ...     return particles
    >>> def observation_func(state):
    ...     # Identity observation model
    ...     return state
    >>> # Create data
    >>> true_states = np.cumsum(np.random.normal(0, 0.1, 100))
    >>> observations = true_states + np.random.normal(0, 0.1, 100)
    >>> # Apply bootstrap particle filter
    >>> filtered_states, weights = bootstrap_particle_filter(
    ...     observations, state_transition, observation_func,
    ...     process_noise_std=0.1, observation_noise_std=0.1, n_particles=1000
    ... )
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    n = len(data)
    
    # Initialize particles
    if initial_state_mean is None:
        initial_state_mean = data[0]
    
    particles = np.random.normal(initial_state_mean, initial_state_std, n_particles)
    
    # Initialize output arrays
    filtered_states = np.zeros(n)
    all_weights = np.zeros((n, n_particles))
    
    # Define process noise function
    def process_noise_func(particles):
        return particles + np.random.normal(0, process_noise_std, particles.shape)
    
    # Define observation likelihood function
    def observation_likelihood_func(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / observation_noise_std) ** 2)
    
    # Apply bootstrap particle filter
    for i in range(n):
        # Propagate particles through state transition model
        particles = state_transition_func(particles)
        
        # Add process noise
        particles = process_noise_func(particles)
        
        # Calculate predicted observations
        predicted_observations = observation_func(particles)
        
        # Calculate weights based on observation likelihood
        weights = observation_likelihood_func(data[i], predicted_observations)
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all weights are zero, reset to uniform
            weights = np.ones(n_particles) / n_particles
        
        # Store weights
        all_weights[i] = weights
        
        # Calculate filtered state (weighted average of particles)
        filtered_states[i] = np.sum(particles * weights)
        
        # Resample (bootstrap)
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        particles = particles[indices]
    
    return filtered_states, all_weights 