"""
Particle Filters Module

This module provides implementations of particle filtering algorithms
for financial time series data.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Tuple, Optional, Union, Callable, List

ParticleFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
LikelihoodFunction = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
InitialStateFunction = Callable[[int], NDArray[np.float64]]

def particle_filter(
    data: ArrayLike,
    state_transition_func: ParticleFunction,
    observation_func: ParticleFunction,
    process_noise_func: ParticleFunction,
    observation_likelihood_func: LikelihoodFunction,
    n_particles: int = 100,
    initial_state_func: Optional[InitialStateFunction] = None,
    resample_threshold: float = 0.5
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply a particle filter to a time series.
    
    The particle filter is a sequential Monte Carlo method that uses a set of particles
    (samples) to represent the posterior distribution of some stochastic process
    given noisy and/or partial observations.
    
    Parameters
    ----------
    data : array_like
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
        
    """
    data_array = np.asarray(data, dtype=np.float64)
    n = len(data_array)
    
    if initial_state_func is None:
        first_observation = float(data_array[0])
        particles = np.random.normal(first_observation, 1.0, n_particles)
    else:
        particles = initial_state_func(n_particles)
    
    weights = np.ones(n_particles, dtype=np.float64) / n_particles
    
    filtered_states = np.zeros(n, dtype=np.float64)
    all_weights = np.zeros((n, n_particles), dtype=np.float64)
    
    for i in range(n):
        particles = state_transition_func(particles)
        
        particles = process_noise_func(particles)
        
        predicted_observations = observation_func(particles)
        
        likelihood = observation_likelihood_func(data_array[i], predicted_observations)
        weights = weights * likelihood
        
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_particles, dtype=np.float64) / n_particles
        
        all_weights[i] = weights
        
        n_eff = 1.0 / np.sum(weights ** 2)
        
        if n_eff / n_particles < resample_threshold:
            indices = np.random.choice(n_particles, size=n_particles, p=weights)
            particles = particles[indices]
            weights = np.ones(n_particles, dtype=np.float64) / n_particles
        
        filtered_states[i] = np.sum(particles * weights)
    
    return filtered_states, all_weights

def bootstrap_particle_filter(
    data: ArrayLike,
    state_transition_func: ParticleFunction,
    observation_func: ParticleFunction,
    process_noise_std: float = 0.1,
    observation_noise_std: float = 0.1,
    n_particles: int = 100,
    initial_state_mean: Optional[float] = None,
    initial_state_std: float = 1.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply a bootstrap particle filter to a time series.
    
    The bootstrap particle filter is a simplified version of the particle filter
    that resamples at every step and uses the state transition prior as the proposal.
    
    Parameters
    ----------
    data : array_like
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
        
    """
    data_array = np.asarray(data, dtype=np.float64)
    n = len(data_array)
    
    if initial_state_mean is None:
        initial_state_mean = float(data_array[0])
    
    particles = np.random.normal(initial_state_mean, initial_state_std, n_particles)
    
    filtered_states = np.zeros(n, dtype=np.float64)
    all_weights = np.zeros((n, n_particles), dtype=np.float64)
    
    def process_noise_func(particles):
        return particles + np.random.normal(0, process_noise_std, particles.shape)
    
    def observation_likelihood_func(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / observation_noise_std) ** 2)
    
    for i in range(n):
        particles = state_transition_func(particles)
        
        particles = process_noise_func(particles)
        
        predicted_observations = observation_func(particles)
        
        weights = observation_likelihood_func(data_array[i], predicted_observations)
        
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_particles, dtype=np.float64) / n_particles
        
        all_weights[i] = weights
        
        filtered_states[i] = np.sum(particles * weights)
        
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        particles = particles[indices]
    
    return filtered_states, all_weights 