# Filters

Pypulate provides a comprehensive set of filtering techniques for financial time series data. This page explains the different types of filters available and how to use them.

## Overview

Filters in Pypulate are designed to clean, smooth, and extract meaningful information from noisy financial time series data. The module includes:

1. **Kalman Filters**: Optimal estimators for linear systems
2. **Signal Filters**: Classical signal processing filters
3. **Adaptive Filters**: Filters that adapt to changing data characteristics
4. **Particle Filters**: Monte Carlo methods for non-linear/non-Gaussian systems

## Kalman Filters

Kalman filters are optimal estimators that infer parameters of interest from indirect, inaccurate, and uncertain observations.

### Standard Kalman Filter

The standard Kalman filter is ideal for linear systems with Gaussian noise:

```python
import numpy as np
from pypulate.filters import kalman_filter

# Create noisy data
x = np.linspace(0, 10, 100)
true_signal = np.sin(x)
noisy_signal = true_signal + np.random.normal(0, 0.1, len(x))

# Apply Kalman filter
filtered_signal = kalman_filter(
    noisy_signal,
    process_variance=1e-5,
    measurement_variance=1e-3
)
```

### Extended Kalman Filter (EKF)

The EKF is used for non-linear systems by linearizing around the current estimate:

```python
import numpy as np
from pypulate.filters import extended_kalman_filter

# Define non-linear system
def state_transition(x):
    # Non-linear state transition function
    return np.array([x[0] + x[1], 0.5 * x[1]])

def observation(x):
    # Non-linear observation function
    return np.array([np.sin(x[0])])

def process_jacobian(x):
    # Jacobian of state transition function
    return np.array([[1, 1], [0, 0.5]])

def observation_jacobian(x):
    # Jacobian of observation function
    return np.array([[np.cos(x[0]), 0]])

# Apply EKF
Q = np.eye(2) * 0.01  # Process noise covariance
R = np.array([[0.1]])  # Observation noise covariance
filtered_states = extended_kalman_filter(
    observations, state_transition, observation,
    process_jacobian, observation_jacobian, Q, R
)
```

### Unscented Kalman Filter (UKF)

The UKF uses sigma points to handle non-linearities without requiring Jacobians:

```python
import numpy as np
from pypulate.filters import unscented_kalman_filter

# Define non-linear system
def state_transition(x):
    # Non-linear state transition function
    return np.array([x[0] + x[1], 0.5 * x[1]])

def observation(x):
    # Non-linear observation function
    return np.array([np.sin(x[0])])

# Apply UKF
Q = np.eye(2) * 0.01  # Process noise covariance
R = np.array([[0.1]])  # Observation noise covariance
filtered_states = unscented_kalman_filter(
    observations, state_transition, observation, Q, R
)
```

## Signal Filters

Signal filters are used to remove noise and extract specific frequency components from time series data.

### Butterworth Filter

The Butterworth filter provides a flat frequency response in the passband:

```python
import numpy as np
from pypulate.filters import butterworth_filter

# Create noisy data with multiple frequency components
x = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * 0.05 * x) + 0.5 * np.sin(2 * np.pi * 0.25 * x)

# Apply lowpass filter to remove high frequency component
filtered = butterworth_filter(
    signal,
    cutoff=0.1,  # Cutoff frequency
    order=4,     # Filter order
    filter_type='lowpass'
)
```

### Savitzky-Golay Filter

The Savitzky-Golay filter smooths data by fitting successive sub-sets of adjacent data points with a low-degree polynomial:

```python
import numpy as np
from pypulate.filters import savitzky_golay_filter

# Create noisy data
x = np.linspace(0, 10, 100)
signal = np.sin(x) + np.random.normal(0, 0.1, len(x))

# Apply Savitzky-Golay filter
filtered = savitzky_golay_filter(
    signal,
    window_length=11,  # Must be odd
    polyorder=3        # Polynomial order
)
```

### Median Filter

The median filter is excellent for removing outliers:

```python
import numpy as np
from pypulate.filters import median_filter

# Create data with outliers
x = np.linspace(0, 10, 100)
signal = np.sin(x)
signal[10] = 5  # Add outlier
signal[50] = -5  # Add outlier

# Apply median filter
filtered = median_filter(signal, kernel_size=5)
```

### Hampel Filter

The Hampel filter is specifically designed for outlier detection and removal:

```python
import numpy as np
from pypulate.filters import hampel_filter

# Create data with outliers
x = np.linspace(0, 10, 100)
signal = np.sin(x)
signal[10] = 5  # Add outlier
signal[50] = -5  # Add outlier

# Apply Hampel filter
filtered = hampel_filter(
    signal,
    window_size=5,  # Window size
    n_sigmas=3.0    # Threshold for outlier detection
)
```

### Hodrick-Prescott Filter

The Hodrick-Prescott filter decomposes a time series into trend and cycle components:

```python
import numpy as np
from pypulate.filters import hodrick_prescott_filter

# Create data with trend and cycle
x = np.linspace(0, 10, 100)
trend = 0.1 * x**2
cycle = np.sin(2 * np.pi * 0.1 * x)
data = trend + cycle

# Apply Hodrick-Prescott filter
trend_component, cycle_component = hodrick_prescott_filter(
    data,
    lambda_param=100  # Smoothing parameter
)
```

## Adaptive Filters

Adaptive filters automatically adjust their parameters based on the input data.

### Adaptive Kalman Filter

The adaptive Kalman filter adjusts its noise parameters based on the observed data:

```python
import numpy as np
from pypulate.filters import adaptive_kalman_filter

# Create noisy data with changing dynamics
x = np.linspace(0, 10, 200)
true_signal = np.sin(x) + 0.1 * x
noise_level = 0.1 * (1 + np.sin(x/2))  # Changing noise level
noisy_signal = true_signal + noise_level * np.random.randn(len(x))

# Apply adaptive Kalman filter
filtered_signal = adaptive_kalman_filter(
    noisy_signal,
    adaptation_rate=0.05,  # Rate of adaptation
    window_size=10         # Window for innovation estimation
)
```

### Least Mean Squares (LMS) Filter

The LMS filter is a simple adaptive filter that minimizes the mean square error:

```python
import numpy as np
from pypulate.filters import least_mean_squares_filter

# Create noisy data
x = np.linspace(0, 10, 1000)
clean_signal = np.sin(2 * np.pi * 0.05 * x)
noise = 0.2 * np.random.randn(len(x))
noisy_signal = clean_signal + noise

# Apply LMS filter
filtered_signal, weights = least_mean_squares_filter(
    noisy_signal,
    filter_length=10,  # Filter length
    mu=0.02            # Step size
)
```

## Particle Filters

Particle filters are Monte Carlo methods that can handle non-linear and non-Gaussian systems.

### Standard Particle Filter

The particle filter uses a set of particles to represent the posterior distribution:

```python
import numpy as np
from pypulate.filters import particle_filter

# Define model functions
def state_transition(particles):
    # Simple random walk model
    return particles

def process_noise(particles):
    # Add Gaussian noise
    return particles + np.random.normal(0, 0.1, particles.shape)

def observation_func(state):
    # Identity observation model
    return state

def observation_likelihood(observation, predicted_observation):
    # Gaussian likelihood
    return np.exp(-0.5 * ((observation - predicted_observation) / 0.1) ** 2)

def initial_state(n):
    # Initial particles from normal distribution
    return np.random.normal(0, 1, n)

# Apply particle filter
filtered_states, weights = particle_filter(
    observations,
    state_transition,
    observation_func,
    process_noise,
    observation_likelihood,
    n_particles=1000,
    initial_state_func=initial_state
)
```

### Bootstrap Particle Filter

The bootstrap particle filter is a simplified version that resamples at every step:

```python
import numpy as np
from pypulate.filters import bootstrap_particle_filter

# Define model functions
def state_transition(particles):
    # Simple random walk model
    return particles

def observation_func(state):
    # Identity observation model
    return state

# Apply bootstrap particle filter
filtered_states, weights = bootstrap_particle_filter(
    observations,
    state_transition,
    observation_func,
    process_noise_std=0.1,
    observation_noise_std=0.1,
    n_particles=1000
)
```

## Choosing the Right Filter

The choice of filter depends on your specific application:

- **Kalman Filters**: Best for linear systems or when you have a good model of the system dynamics
- **Signal Filters**: Good for general noise removal and frequency-based filtering
- **Adaptive Filters**: Useful when the signal characteristics change over time
- **Particle Filters**: Best for highly non-linear systems or non-Gaussian noise

For financial time series, consider:

- **Kalman/Adaptive Filters**: For tracking changing trends
- **Hampel/Median Filters**: For removing outliers (e.g., flash crashes)
- **Hodrick-Prescott Filter**: For separating trend and cycle components
- **Butterworth Filter**: For removing high-frequency noise

## Example: Combining Filters

You can combine multiple filters for more sophisticated processing:

```python
import numpy as np
import matplotlib.pyplot as plt
from pypulate.filters import hampel_filter, kalman_filter

# Create data with outliers and noise
x = np.linspace(0, 10, 200)
true_signal = np.sin(x) + 0.1 * x
noisy_signal = true_signal + 0.1 * np.random.randn(len(x))
noisy_signal[20] = 5  # Add outlier
noisy_signal[100] = -5  # Add outlier

# First remove outliers with Hampel filter
outlier_removed = hampel_filter(noisy_signal, window_size=5, n_sigmas=3.0)

# Then smooth with Kalman filter
final_signal = kalman_filter(outlier_removed, process_variance=1e-5, measurement_variance=1e-3)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(x, true_signal, 'k-', label='True Signal')
plt.plot(x, noisy_signal, 'r.', alpha=0.5, label='Noisy Signal with Outliers')
plt.plot(x, outlier_removed, 'g-', alpha=0.7, label='After Hampel Filter')
plt.plot(x, final_signal, 'b-', linewidth=2, label='After Kalman Filter')
plt.legend()
plt.title('Multi-stage Filtering')
plt.grid(True, alpha=0.3)
plt.show()
``` 

Filters also works with Parray chain methods. You can combine them with other tools easily to make advanced techniques.