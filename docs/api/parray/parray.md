---
title: Parray API
---

# Parray API Reference

This page documents the API for the `Parray` class in Pypulate.

## Class Definition

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: true
      show_source: false
      members:
        - __new__
        - __array_finalize__

## Performance Optimization

### GPU Acceleration

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - is_gpu_available
        - get_gpu_info
        - enable_gpu
        - disable_gpu
        - gpu

### Parallel Processing

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - enable_parallel
        - disable_parallel
        - parallel

### Memory Optimization

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - optimize_memory
        - disable_memory_optimization
        - to_chunks
        - from_chunks

## Moving Averages

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - sma
        - ema
        - wma
        - tma
        - smma
        - zlma
        - hma
        - kama
        - t3
        - frama
        - mcginley_dynamic

## Momentum Indicators

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - momentum
        - roc
        - percent_change
        - difference
        - rsi
        - macd
        - stochastic_oscillator
        - tsi
        - williams_r
        - cci
        - adx

## Volatility Indicators

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - historical_volatility
        - atr
        - bollinger_bands
        - keltner_channels
        - donchian_channels
        - volatility_ratio
        - typical_price

## Statistical Functions

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - rolling_max
        - rolling_min
        - rolling_std
        - rolling_var
        - zscore
        - descriptive_stats
        - correlation_matrix
        - covariance_matrix
        - autocorrelation
        - partial_autocorrelation

## Stationarity Tests

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - jarque_bera_test
        - augmented_dickey_fuller_test
        - kpss_test
        - ljung_box_test
        - durbin_watson_test
        - arch_test
        - kolmogorov_smirnov_test
        - hurst_exponent
        - variance_ratio_test
        - granger_causality_test

## Signal Generation

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - crossover
        - crossunder
        - wave
        - zigzag
        - slope

## Filters and Smoothing

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - kalman_filter
        - adaptive_kalman_filter
        - butterworth_filter
        - savitzky_golay_filter
        - hampel_filter
        - hodrick_prescott_filter

## Data Transformations

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - normalize
        - normalize_l1
        - normalize_l2
        - standardize
        - min_max_scale
        - robust_scale
        - quantile_transform
        - winsorize
        - remove_outliers
        - fill_missing
        - interpolate_missing
        - log_transform
        - power_transform
        - scale_to_range
        - clip_outliers
        - discretize
        - polynomial_features
        - resample
        - dynamic_tanh

## Time Series Operations

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - lag_features
        - rolling_window
        - rolling_statistics
        - rolling_apply

## Custom Processing

::: pypulate.dtypes.parray.Parray
    options:
      show_root_heading: false
      show_source: false
      members:
        - apply
        - apply_along_axis 