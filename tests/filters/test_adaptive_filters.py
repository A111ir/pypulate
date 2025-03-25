import pytest
import numpy as np
from pypulate.filters.adaptive_filters import (
    adaptive_kalman_filter,
    least_mean_squares_filter,
    recursive_least_squares_filter
)


def test_adaptive_kalman_filter_basic():
    # Test with simple constant signal
    data = np.ones(100)
    filtered = adaptive_kalman_filter(data)
    
    # Filter should approximately recover the constant signal
    assert np.allclose(filtered, 1.0, atol=1e-3)
    assert filtered.shape == data.shape


def test_adaptive_kalman_filter_noisy_data():
    # Create a sine wave with noise
    x = np.linspace(0, 10, 200)
    true_signal = np.sin(x)
    noise = 0.1 * np.random.randn(len(x))
    noisy_signal = true_signal + noise
    
    # Apply filter
    filtered = adaptive_kalman_filter(noisy_signal, adaptation_rate=0.05)
    
    # Filter should reduce noise (MSE should be lower)
    noisy_mse = np.mean((noisy_signal - true_signal) ** 2)
    filtered_mse = np.mean((filtered - true_signal) ** 2)
    
    assert filtered_mse < noisy_mse
    assert filtered.shape == noisy_signal.shape


def test_adaptive_kalman_filter_parameters():
    # Test with different parameter values
    data = np.random.randn(100)
    
    # Default parameters
    filtered1 = adaptive_kalman_filter(data)
    
    # Different process variance
    filtered2 = adaptive_kalman_filter(data, process_variance_init=1e-2)
    
    # Different measurement variance
    filtered3 = adaptive_kalman_filter(data, measurement_variance_init=1e-1)
    
    # Different adaptation rate
    filtered4 = adaptive_kalman_filter(data, adaptation_rate=0.1)
    
    # Different window size
    filtered5 = adaptive_kalman_filter(data, window_size=20)
    
    # Different initial state
    filtered6 = adaptive_kalman_filter(data, initial_state=0.0)
    
    # Different initial covariance
    filtered7 = adaptive_kalman_filter(data, initial_covariance=2.0)
    
    # All should return arrays of the same shape
    assert filtered1.shape == data.shape
    assert filtered2.shape == data.shape
    assert filtered3.shape == data.shape
    assert filtered4.shape == data.shape
    assert filtered5.shape == data.shape
    assert filtered6.shape == data.shape
    assert filtered7.shape == data.shape
    
    # Different parameters should produce different results
    assert not np.allclose(filtered1, filtered2)
    assert not np.allclose(filtered1, filtered3)
    assert not np.allclose(filtered1, filtered4)


def test_adaptive_kalman_filter_changing_dynamics():
    # Test with signal that has changing dynamics
    x = np.linspace(0, 10, 200)
    signal = np.zeros_like(x)
    
    # Piecewise signal with changing dynamics
    signal[:50] = 1.0  # Constant
    signal[50:100] = np.linspace(1.0, 2.0, 50)  # Linear increase
    signal[100:150] = 2.0 + 0.5 * np.sin(3 * (x[100:150] - x[100]))  # Sinusoidal
    signal[150:] = 2.0  # Constant again
    
    # Add noise
    noisy_signal = signal + 0.1 * np.random.randn(len(x))
    
    # Apply filter
    filtered = adaptive_kalman_filter(
        noisy_signal, 
        process_variance_init=1e-3, 
        measurement_variance_init=1e-2,
        adaptation_rate=0.1
    )
    
    # Filter should track the changing dynamics
    mse = np.mean((filtered - signal) ** 2)
    noisy_mse = np.mean((noisy_signal - signal) ** 2)
    
    assert mse < noisy_mse
    
    # Check specific regions - filter should track each regime
    # For the constant regions, filter should be very close to true signal
    assert np.mean((filtered[20:40] - signal[20:40]) ** 2) < 0.02
    assert np.mean((filtered[170:190] - signal[170:190]) ** 2) < 0.02


def test_least_mean_squares_filter_basic():
    # Test with simple constant signal
    data = np.ones(100)
    filtered, weights = least_mean_squares_filter(data)
    
    # Filter should be close to the constant input after settling period
    # (doesn't need to be exactly 1.0)
    assert np.all(filtered[80:] > 0.97)  # Should be approaching 1.0 by the end
    assert filtered.shape == data.shape
    assert len(weights) == 5  # Default filter length


def test_least_mean_squares_filter_parameters():
    # Test with different parameter values
    data = np.random.randn(100)
    
    # Default parameters
    filtered1, weights1 = least_mean_squares_filter(data)
    
    # Different filter length
    filtered2, weights2 = least_mean_squares_filter(data, filter_length=10)
    
    # Different learning rate
    filtered3, weights3 = least_mean_squares_filter(data, mu=0.05)
    
    # Different initial weights
    initial_weights = np.ones(5) * 0.2
    filtered4, weights4 = least_mean_squares_filter(data, initial_weights=initial_weights)
    
    # With explicit desired signal
    desired = np.zeros_like(data)
    filtered5, weights5 = least_mean_squares_filter(data, desired=desired)
    
    # All should return arrays of the same shape
    assert filtered1.shape == data.shape
    assert filtered2.shape == data.shape
    assert filtered3.shape == data.shape
    assert filtered4.shape == data.shape
    assert filtered5.shape == data.shape
    
    # Check weights dimensions
    assert len(weights1) == 5
    assert len(weights2) == 10
    assert len(weights3) == 5
    assert len(weights4) == 5
    assert len(weights5) == 5
    
    # Different parameters should produce different results
    assert not np.allclose(filtered1, filtered2)
    assert not np.allclose(filtered1, filtered3)
    assert not np.allclose(filtered1, filtered4)
    assert not np.allclose(filtered1, filtered5)


def test_least_mean_squares_filter_noise_reduction():
    # Create a sine wave with noise
    x = np.linspace(0, 10, 500)
    clean_signal = np.sin(2 * np.pi * 0.05 * x)
    noise = 0.2 * np.random.randn(len(x))
    noisy_signal = clean_signal + noise
    
    # Apply LMS filter
    filtered, weights = least_mean_squares_filter(
        noisy_signal, 
        filter_length=20, 
        mu=0.02
    )
    
    # Filter should reduce noise (MSE should be lower in the middle portion
    # after filter has adapted)
    start_idx = 50  # Skip initial transient
    noisy_mse = np.mean((noisy_signal[start_idx:] - clean_signal[start_idx:]) ** 2)
    filtered_mse = np.mean((filtered[start_idx:] - clean_signal[start_idx:]) ** 2)
    
    assert filtered_mse < noisy_mse


def test_recursive_least_squares_filter_basic():
    # Test with simple constant signal
    data = np.ones(100)
    filtered, weights = recursive_least_squares_filter(data)
    
    # Filter should be close to the constant input after settling period
    # (doesn't need to be exactly 1.0)
    assert np.all(filtered[80:] > 0.98)  # Should be very close to 1.0 by the end
    assert filtered.shape == data.shape
    assert len(weights) == 5  # Default filter length


def test_recursive_least_squares_filter_parameters():
    # Test with different parameter values
    data = np.random.randn(100)
    
    # Default parameters
    filtered1, weights1 = recursive_least_squares_filter(data)
    
    # Different filter length
    filtered2, weights2 = recursive_least_squares_filter(data, filter_length=10)
    
    # Different forgetting factor
    filtered3, weights3 = recursive_least_squares_filter(data, forgetting_factor=0.95)
    
    # Different delta parameter
    filtered4, weights4 = recursive_least_squares_filter(data, delta=0.5)
    
    # Different initial weights
    initial_weights = np.ones(5) * 0.2
    filtered5, weights5 = recursive_least_squares_filter(data, initial_weights=initial_weights)
    
    # With explicit desired signal
    desired = np.zeros_like(data)
    filtered6, weights6 = recursive_least_squares_filter(data, desired=desired)
    
    # All should return arrays of the same shape
    assert filtered1.shape == data.shape
    assert filtered2.shape == data.shape
    assert filtered3.shape == data.shape
    assert filtered4.shape == data.shape
    assert filtered5.shape == data.shape
    assert filtered6.shape == data.shape
    
    # Check weights dimensions
    assert len(weights1) == 5
    assert len(weights2) == 10
    assert len(weights3) == 5
    assert len(weights4) == 5
    assert len(weights5) == 5
    assert len(weights6) == 5
    
    # Different parameters should produce different results
    assert not np.allclose(filtered1, filtered2)
    assert not np.allclose(filtered1, filtered3)
    assert not np.allclose(filtered1, filtered4)
    assert not np.allclose(filtered1, filtered5)
    assert not np.allclose(filtered1, filtered6)


def test_recursive_least_squares_filter_noise_reduction():
    # Test with a simple step function rather than a sine wave
    # This is easier for the filter to track
    n = 200
    clean_signal = np.zeros(n)
    clean_signal[50:] = 1.0  # Step function
    
    # Add noise
    np.random.seed(42)
    noise = 0.3 * np.random.randn(n)
    noisy_signal = clean_signal + noise
    
    # Apply RLS filter
    filtered, weights = recursive_least_squares_filter(
        noisy_signal, 
        filter_length=10,
        forgetting_factor=0.97
    )
    
    # Verify filter converges to steady state after step
    # Skip initial part and transient after step
    steady_region = slice(100, 150)
    
    # Calculate variance of filtered vs noisy in steady region
    noisy_var = np.var(noisy_signal[steady_region])
    filtered_var = np.var(filtered[steady_region])
    
    # Filtered signal should have less variance (less noise) in steady region
    assert filtered_var < noisy_var
    
    # Mean of filtered signal should be close to true value (1.0) in steady region
    steady_mean = np.mean(filtered[steady_region])
    assert np.abs(steady_mean - 1.0) < 0.1


def test_recursive_vs_lms_convergence():
    # RLS should converge faster than LMS
    x = np.linspace(0, 10, 500)
    clean_signal = np.sin(2 * np.pi * 0.05 * x) + 0.5 * np.sin(2 * np.pi * 0.1 * x)
    noise = 0.2 * np.random.randn(len(x))
    noisy_signal = clean_signal + noise
    
    # Apply both filters with same length
    filter_length = 20
    
    # LMS with moderate learning rate
    lms_filtered, _ = least_mean_squares_filter(
        noisy_signal, 
        filter_length=filter_length, 
        mu=0.01
    )
    
    # RLS with typical parameters
    rls_filtered, _ = recursive_least_squares_filter(
        noisy_signal, 
        filter_length=filter_length, 
        forgetting_factor=0.98
    )
    
    # Check MSE in early adaptation phase
    early_idx = slice(filter_length+10, filter_length+50)
    lms_early_mse = np.mean((lms_filtered[early_idx] - clean_signal[early_idx]) ** 2)
    rls_early_mse = np.mean((rls_filtered[early_idx] - clean_signal[early_idx]) ** 2)
    
    # RLS should have lower error in early adaptation phase
    assert rls_early_mse < lms_early_mse


def test_filters_with_edge_cases():
    # Test with very short signal
    short_data = np.random.randn(10)
    
    # All filters should work with shorter signals
    akf_result = adaptive_kalman_filter(short_data)
    lms_result, _ = least_mean_squares_filter(short_data, filter_length=3)
    rls_result, _ = recursive_least_squares_filter(short_data, filter_length=3)
    
    assert len(akf_result) == len(short_data)
    assert len(lms_result) == len(short_data)
    assert len(rls_result) == len(short_data)
    
    # Test with constant input
    const_data = np.zeros(50)
    
    akf_const = adaptive_kalman_filter(const_data)
    lms_const, _ = least_mean_squares_filter(const_data)
    rls_const, _ = recursive_least_squares_filter(const_data)
    
    # All filters should preserve the constant signal
    assert np.allclose(akf_const[10:], 0, atol=1e-5)
    assert np.allclose(lms_const[10:], 0, atol=1e-5)
    assert np.allclose(rls_const[10:], 0, atol=1e-5) 