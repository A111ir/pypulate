"""
Tests for the particle_filters module.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from typing import Tuple, Callable

from pypulate.filters.particle_filters import particle_filter, bootstrap_particle_filter


def generate_random_walk_data(n_steps: int = 100, process_std: float = 0.1, 
                             observation_std: float = 0.1, 
                             initial_state: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a random walk with noisy observations for testing."""
    true_states = np.zeros(n_steps, dtype=np.float64)
    true_states[0] = initial_state
    
    # Generate process noise
    process_noise = np.random.normal(0, process_std, n_steps-1)
    
    # Generate true states using random walk
    for i in range(1, n_steps):
        true_states[i] = true_states[i-1] + process_noise[i-1]
    
    # Generate observations with noise
    observations = true_states + np.random.normal(0, observation_std, n_steps)
    
    return true_states, observations


def test_particle_filter_basic():
    """Test basic functionality of the particle filter."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=50)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    def process_noise(particles):
        return particles + np.random.normal(0, 0.1, particles.shape)
    
    def observation_likelihood(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / 0.1) ** 2)
    
    # Apply particle filter
    filtered_states, weights = particle_filter(
        observations, state_transition, observation_func,
        process_noise, observation_likelihood, n_particles=100
    )
    
    # Check shapes
    assert filtered_states.shape == true_states.shape
    assert weights.shape == (len(observations), 100)
    
    # Check if filtered states track true states reasonably well
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(filtered_states - true_states))
    # MAE should be lower than the observation noise
    assert mae < 0.15


def test_particle_filter_custom_initial():
    """Test particle filter with custom initial state function."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=50, initial_state=5.0)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    def process_noise(particles):
        return particles + np.random.normal(0, 0.1, particles.shape)
    
    def observation_likelihood(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / 0.1) ** 2)
    
    def initial_state(n):
        return np.random.normal(5.0, 0.5, n)
    
    # Apply particle filter
    filtered_states, weights = particle_filter(
        observations, state_transition, observation_func,
        process_noise, observation_likelihood, n_particles=100,
        initial_state_func=initial_state
    )
    
    # Check if filtered states track true states reasonably well
    mae = np.mean(np.abs(filtered_states - true_states))
    assert mae < 0.15


def test_particle_filter_nonlinear():
    """Test particle filter with non-linear observation model."""
    # Generate test data
    n_steps = 50
    true_states = np.linspace(0, np.pi, n_steps)  # True states from 0 to pi
    observations = np.sin(true_states) + np.random.normal(0, 0.1, n_steps)  # Sin function with noise
    
    # Define model functions
    def state_transition(particles):
        return particles + np.pi / (n_steps - 1)  # Increment by step size
    
    def observation_func(state):
        return np.sin(state)  # Non-linear observation function
    
    def process_noise(particles):
        return particles + np.random.normal(0, 0.05, particles.shape)
    
    def observation_likelihood(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / 0.1) ** 2)
    
    # Apply particle filter
    filtered_states, weights = particle_filter(
        observations, state_transition, observation_func,
        process_noise, observation_likelihood, n_particles=200
    )
    
    # Check if filtered states track true states reasonably well
    # Due to non-linearity, we use a larger tolerance
    mae = np.mean(np.abs(filtered_states - true_states))
    assert mae < 0.5


def test_particle_filter_resampling():
    """Test particle filter with different resampling thresholds."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=50)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    def process_noise(particles):
        return particles + np.random.normal(0, 0.1, particles.shape)
    
    def observation_likelihood(observation, predicted_observations):
        # More peaked likelihood to encourage resampling
        return np.exp(-2.0 * ((observation - predicted_observations) / 0.1) ** 2)
    
    # Apply particle filter with low resampling threshold (frequent resampling)
    filtered_states_low, _ = particle_filter(
        observations, state_transition, observation_func,
        process_noise, observation_likelihood, n_particles=100,
        resample_threshold=0.3
    )
    
    # Apply particle filter with high resampling threshold (infrequent resampling)
    filtered_states_high, _ = particle_filter(
        observations, state_transition, observation_func,
        process_noise, observation_likelihood, n_particles=100,
        resample_threshold=0.9
    )
    
    # Both should track reasonably well, but might have different characteristics
    # Just check that they're not identical, which would indicate resampling isn't happening differently
    assert not np.allclose(filtered_states_low, filtered_states_high, atol=1e-3)


def test_bootstrap_particle_filter_basic():
    """Test basic functionality of the bootstrap particle filter."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=50)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    # Apply bootstrap particle filter
    filtered_states, weights = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.1, observation_noise_std=0.1, n_particles=100
    )
    
    # Check shapes
    assert filtered_states.shape == true_states.shape
    assert weights.shape == (len(observations), 100)
    
    # Check if filtered states track true states reasonably well
    mae = np.mean(np.abs(filtered_states - true_states))
    assert mae < 0.15


def test_bootstrap_particle_filter_custom_initial():
    """Test bootstrap particle filter with custom initial state."""
    # Generate test data starting from state 5.0
    true_states, observations = generate_random_walk_data(n_steps=50, initial_state=5.0)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    # Apply bootstrap particle filter with specified initial state mean
    filtered_states, weights = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.1, observation_noise_std=0.1, n_particles=100,
        initial_state_mean=5.0, initial_state_std=0.5
    )
    
    # Check if filtered states track true states reasonably well
    mae = np.mean(np.abs(filtered_states - true_states))
    assert mae < 0.15


def test_bootstrap_particle_filter_nonlinear():
    """Test bootstrap particle filter with non-linear observation model."""
    # Generate test data
    n_steps = 50
    true_states = np.linspace(0, np.pi, n_steps)  # True states from 0 to pi
    observations = np.sin(true_states) + np.random.normal(0, 0.1, n_steps)  # Sin function with noise
    
    # Define model functions
    def state_transition(particles):
        return particles + np.pi / (n_steps - 1)  # Increment by step size
    
    def observation_func(state):
        return np.sin(state)  # Non-linear observation function
    
    # Apply bootstrap particle filter
    filtered_states, weights = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.05, observation_noise_std=0.1, n_particles=200
    )
    
    # Check if filtered states track true states reasonably well
    # Due to non-linearity, we use a larger tolerance
    mae = np.mean(np.abs(filtered_states - true_states))
    assert mae < 0.5


def test_bootstrap_particle_filter_process_noise():
    """Test bootstrap particle filter with different process noise levels."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=50, process_std=0.2)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    # Apply bootstrap particle filter with small process noise
    filtered_states_small, _ = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.05, observation_noise_std=0.1, n_particles=100
    )
    
    # Apply bootstrap particle filter with large process noise
    filtered_states_large, _ = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.3, observation_noise_std=0.1, n_particles=100
    )
    
    # Both should track, but with different characteristics
    assert not np.allclose(filtered_states_small, filtered_states_large, atol=1e-3)


def test_bootstrap_particle_filter_observation_noise():
    """Test bootstrap particle filter with different observation noise parameters."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=50, observation_std=0.2)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    # Apply bootstrap particle filter with accurate observation noise
    filtered_states_accurate, _ = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.1, observation_noise_std=0.2, n_particles=100
    )
    
    # Apply bootstrap particle filter with incorrect observation noise
    filtered_states_inaccurate, _ = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.1, observation_noise_std=0.05, n_particles=100
    )
    
    # The one with correct noise should track better
    mae_accurate = np.mean(np.abs(filtered_states_accurate - true_states))
    mae_inaccurate = np.mean(np.abs(filtered_states_inaccurate - true_states))
    
    # Since these are stochastic tests, we use a higher tolerance
    # In rare cases this might fail due to random chance
    assert mae_accurate <= mae_inaccurate * 1.5


def test_particle_filter_convergence():
    """Test particle filter convergence with increasing number of particles."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=30)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    def process_noise(particles):
        return particles + np.random.normal(0, 0.1, particles.shape)
    
    def observation_likelihood(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / 0.1) ** 2)
    
    # Track errors with different numbers of particles
    n_particles_list = [10, 50, 200]
    errors = []
    
    for n_particles in n_particles_list:
        filtered_states, _ = particle_filter(
            observations, state_transition, observation_func,
            process_noise, observation_likelihood, n_particles=n_particles
        )
        
        mae = np.mean(np.abs(filtered_states - true_states))
        errors.append(mae)
    
    # Errors should generally decrease with more particles, but stochastic nature may cause exceptions
    # We just check that the highest error is with the fewest particles
    assert np.argmax(errors) == 0


def test_edge_cases():
    """Test edge cases for both filters."""
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    def process_noise(particles):
        return particles + np.random.normal(0, 0.1, particles.shape)
    
    def observation_likelihood(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / 0.1) ** 2)
    
    # Test with single observation
    single_observation = np.array([1.0])
    
    filtered_states, weights = particle_filter(
        single_observation, state_transition, observation_func,
        process_noise, observation_likelihood, n_particles=10
    )
    
    assert filtered_states.shape == (1,)
    assert weights.shape == (1, 10)
    
    # Test bootstrap filter with single observation
    bootstrap_states, bootstrap_weights = bootstrap_particle_filter(
        single_observation, state_transition, observation_func,
        process_noise_std=0.1, observation_noise_std=0.1, n_particles=10
    )
    
    assert bootstrap_states.shape == (1,)
    assert bootstrap_weights.shape == (1, 10)


def test_compare_filters():
    """Compare particle filter and bootstrap particle filter."""
    # Generate test data
    true_states, observations = generate_random_walk_data(n_steps=50)
    
    # Define model functions
    def state_transition(particles):
        return particles
    
    def observation_func(state):
        return state
    
    def process_noise(particles):
        return particles + np.random.normal(0, 0.1, particles.shape)
    
    def observation_likelihood(observation, predicted_observations):
        return np.exp(-0.5 * ((observation - predicted_observations) / 0.1) ** 2)
    
    # Apply particle filter
    filtered_states, _ = particle_filter(
        observations, state_transition, observation_func,
        process_noise, observation_likelihood, n_particles=100
    )
    
    # Apply bootstrap particle filter
    bootstrap_states, _ = bootstrap_particle_filter(
        observations, state_transition, observation_func,
        process_noise_std=0.1, observation_noise_std=0.1, n_particles=100
    )
    
    # Both should track true states reasonably well
    mae_pf = np.mean(np.abs(filtered_states - true_states))
    mae_bpf = np.mean(np.abs(bootstrap_states - true_states))
    
    assert mae_pf < 0.2
    assert mae_bpf < 0.2
    
    # The results should be somewhat different but comparable
    assert not np.allclose(filtered_states, bootstrap_states, atol=1e-3)
    # MAE difference shouldn't be huge
    assert abs(mae_pf - mae_bpf) < 0.1 