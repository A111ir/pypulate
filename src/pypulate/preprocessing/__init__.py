"""
Data Processing Module

This module provides data processing utilities for financial data analysis
without dependencies on pandas, using only numpy and scipy.
"""

from .preprocessing import (
    normalize, standardize, winsorize, remove_outliers,
    fill_missing, interpolate_missing, resample, 
    rolling_window, lag_features, difference, log_transform,
    min_max_scale, robust_scale, quantile_transform,
    power_transform, scale_to_range, clip_outliers,
    discretize, polynomial_features, dynamic_tanh
)

from .statistics import (
    descriptive_stats, correlation_matrix, covariance_matrix,
    autocorrelation, partial_autocorrelation, 
    jarque_bera_test, augmented_dickey_fuller_test,
    granger_causality_test, ljung_box_test, kpss_test,
    variance_ratio_test, durbin_watson_test, arch_test,
    kolmogorov_smirnov_test, rolling_statistics, hurst_exponent
)

__all__ = [
    # Preprocessing
    'normalize', 'standardize', 'winsorize', 'remove_outliers',
    'fill_missing', 'interpolate_missing', 'resample',
    'rolling_window', 'lag_features', 'difference', 'log_transform',
    'min_max_scale', 'robust_scale', 'quantile_transform',
    'power_transform', 'scale_to_range', 'clip_outliers',
    'discretize', 'polynomial_features', 'dynamic_tanh',
    
    # Statistics
    'descriptive_stats', 'correlation_matrix', 'covariance_matrix',
    'autocorrelation', 'partial_autocorrelation',
    'jarque_bera_test', 'augmented_dickey_fuller_test',
    'granger_causality_test', 'ljung_box_test', 'kpss_test',
    'variance_ratio_test', 'durbin_watson_test', 'arch_test',
    'kolmogorov_smirnov_test', 'rolling_statistics', 'hurst_exponent'
]
