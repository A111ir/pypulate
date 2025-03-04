"""
Filters Module

This module provides various filtering techniques for financial time series data,
including Kalman filters, moving average filters, and other signal processing filters.
"""

from .kalman import (
    kalman_filter, 
    extended_kalman_filter, 
    unscented_kalman_filter
)

from .signal_filters import (
    butterworth_filter,
    chebyshev_filter,
    savitzky_golay_filter,
    wiener_filter,
    median_filter,
    hampel_filter,
    hodrick_prescott_filter,
    baxter_king_filter
)

from .adaptive_filters import (
    adaptive_kalman_filter,
    least_mean_squares_filter,
    recursive_least_squares_filter
)

from .particle_filters import (
    particle_filter,
    bootstrap_particle_filter
)

__all__ = [
    # Kalman filters
    'kalman_filter',
    'extended_kalman_filter',
    'unscented_kalman_filter',
    
    # Signal filters
    'butterworth_filter',
    'chebyshev_filter',
    'savitzky_golay_filter',
    'wiener_filter',
    'median_filter',
    'hampel_filter',
    'hodrick_prescott_filter',
    'baxter_king_filter',
    
    # Adaptive filters
    'adaptive_kalman_filter',
    'least_mean_squares_filter',
    'recursive_least_squares_filter',
    
    # Particle filters
    'particle_filter',
    'bootstrap_particle_filter'
]
