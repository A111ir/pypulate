# Parray - Enhanced NumPy Arrays for Financial Analysis

The `Parray` class is a powerful extension of NumPy arrays specifically designed for financial time series analysis. It provides a comprehensive set of methods for technical analysis, statistical operations, and performance optimization.

## Introduction

`Parray` inherits from `numpy.ndarray`, so it has all the functionality of NumPy arrays plus a rich ecosystem of financial analysis methods. Its key advantages include:

- Method chaining for concise, readable code
- GPU acceleration for improved performance
- Parallel processing for handling large datasets
- Comprehensive set of financial indicators and statistical tools
- Memory optimization for large datasets

## Creating a Parray

```python
from pypulate.dtypes import Parray
import numpy as np

# From a list
data = [1, 2, 3, 4, 5]
p = Parray(data)

# From a NumPy array
data = np.array([1, 2, 3, 4, 5])
p = Parray(data)

# From a 2D array (e.g., multiple time series)
data_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
p_2d = Parray(data_2d)

# With memory optimization
p_optimized = Parray(data, memory_optimized=True)
```

## Performance Optimization

!!! note "Performance Optimization"
    Most Parray methods are already highly optimized using efficient NumPy operations and vectorization. You typically don't need to enable additional optimization features unless you're working with very large datasets or have specific performance requirements.

### GPU Acceleration

```python
# Check if GPU is available
if Parray.is_gpu_available():
    # Get GPU info
    gpu_info = Parray.get_gpu_info()
    print(f"GPU devices: {len(gpu_info['devices'])}")
    
    # Create a Parray with GPU acceleration
    p = Parray([1, 2, 3, 4, 5])
    p.enable_gpu()
    
    # Perform calculations on GPU
    result = p.standardize()
    
    # Disable GPU when no longer needed
    p.disable_gpu()
```

### Parallel Processing

```python
# Enable parallel processing with default settings
p = Parray(np.random.random(1000000))
p.enable_parallel()

# Enable with custom settings
p.enable_parallel(num_workers=8, chunk_size=100000)

# Perform calculation in parallel
result = p.standardize()

# Disable parallel processing
p.disable_parallel()
```

### Memory Optimization

```python
# Create a memory-optimized Parray
p = Parray(np.random.random(1000000), memory_optimized=True)

# Or optimize an existing Parray
p = Parray(np.random.random(1000000))
p.optimize_memory()

# Process large datasets in chunks
chunks = p.to_chunks(chunk_size=100000)
results = [chunk.normalize() for chunk in chunks]
final_result = Parray.from_chunks(results)

# Disable memory optimization
p.disable_memory_optimization()
```

## Moving Averages

```python
# Sample price data
prices = Parray([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16, 17, 18, 19, 20, 21])

# Simple Moving Average
sma = prices.sma(period=5)

# Exponential Moving Average
ema = prices.ema(period=5)

# Weighted Moving Average
wma = prices.wma(period=5)

# Triple Exponential Moving Average (T3)
t3 = prices.t3(period=5, vfactor=0.7)

# Hull Moving Average
hma = prices.hma(period=5)

# Triangular Moving Average
tma = prices.tma(period=5)

# Smoothed Moving Average
smma = prices.smma(period=5)

# Zero-Lag Moving Average
zlma = prices.zlma(period=5)

# Kaufman Adaptive Moving Average
kama = prices.kama(period=5, fast_period=2, slow_period=30)

# Fractal Adaptive Moving Average
frama = prices.frama(period=5)

# McGinley Dynamic
md = prices.mcginley_dynamic(period=5, k=0.6)
```

## Momentum Indicators

```python
# RSI (Relative Strength Index)
rsi = prices.rsi(period=14)

# Momentum
momentum = prices.momentum(period=14)

# Rate of Change
roc = prices.roc(period=14)

# MACD (Moving Average Convergence Divergence)
# Note: Your data must be longer than the slow_period parameter
# Create sufficiently long data for MACD calculation
longer_prices = Parray(np.arange(1, 51))  # 50 data points
# Or use smaller periods for smaller datasets
macd_line, signal_line, histogram = prices.macd(fast_period=5, slow_period=10, signal_period=3)  # For smaller datasets
# Default parameters (requires at least 26 data points)
macd_line, signal_line, histogram = longer_prices.macd(fast_period=12, slow_period=26, signal_period=9)

# Stochastic Oscillator (requires high and low data)
high = Parray([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 18, 19, 20, 21, 22])
low = Parray([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 17, 16, 15, 16, 17, 18, 19, 20])
k, d = prices.stochastic_oscillator(high, low, k_period=14, d_period=3)

# True Strength Index
# Note: The data length must be greater than the long_period parameter (25 by default)
# For smaller datasets, use smaller parameters
tsi_line, signal_line = prices.tsi(long_period=5, short_period=3, signal_period=2)  # Works with smaller data
# With default parameters (requires at least 25 data points)
tsi_line, signal_line = longer_prices.tsi(long_period=25, short_period=13, signal_period=7)

# Williams %R (requires high and low data)
williams_r = prices.williams_r(high, low, period=14)

# Commodity Channel Index (CCI)
cci = prices.cci(period=20, constant=0.015)

# Average Directional Index (ADX)
adx = prices.adx(period=14)
```

## Volatility Indicators

```python
# Bollinger Bands
upper, middle, lower = prices.bollinger_bands(period=20, std_dev=2.0)

# Average True Range (ATR)
atr = prices.atr(high, low, period=14)

# Keltner Channels
k_upper, k_middle, k_lower = prices.keltner_channels(high=high, low=low, period=20, atr_period=10, multiplier=2.0)

# Donchian Channels
d_upper, d_middle, d_lower = prices.donchian_channels(high=high, low=low, period=20)

# Historical Volatility
hv = prices.historical_volatility(period=21, annualization_factor=252)

# Volatility Ratio
vr = prices.volatility_ratio(period=21, smooth_period=5)
```

## Statistical Functions

```python
# Rolling Maximum
roll_max = prices.rolling_max(period=14)

# Rolling Minimum
roll_min = prices.rolling_min(period=14)

# Rolling Standard Deviation
roll_std = prices.rolling_std(period=14)

# Rolling Variance
roll_var = prices.rolling_var(period=14)

# Z-Score
zscore = prices.zscore(period=14)

# Descriptive Statistics
stats = prices.descriptive_stats()
print(f"Mean: {stats['mean']}, Std: {stats['std']}, Skewness: {stats['skewness']}")

# Autocorrelation
acf = prices.autocorrelation(max_lag=20)

# Partial Autocorrelation
pacf = prices.partial_autocorrelation(max_lag=20)

# Stationarity Tests
jb_stat, jb_pvalue = prices.jarque_bera_test()
adf_stat, adf_pvalue = prices.augmented_dickey_fuller_test()
kpss_stat, kpss_pvalue = prices.kpss_test()

# Correlation Matrix (requires 2D data)
multi_series = Parray(np.random.random((100, 3)))  # 3 series, 100 observations each
corr_matrix = multi_series.correlation_matrix(method='pearson')

# Covariance Matrix
cov_matrix = multi_series.covariance_matrix(ddof=1)
```

## crossovers

```python
# Crossover and Crossunder Detection
fast_ma = prices.ema(5)
slow_ma = prices.ema(20)

# Detect when fast MA crosses above slow MA (buy signal)
buy_signals = fast_ma.crossover(slow_ma)

# Detect when fast MA crosses below slow MA (sell signal)
sell_signals = fast_ma.crossunder(slow_ma)
```

## Filters and Smoothing

```python
# Kalman Filter
kf = prices.kalman_filter(process_variance=1e-5, measurement_variance=1e-3)

# Adaptive Kalman Filter
akf = prices.adaptive_kalman_filter(process_variance_init=1e-5, measurement_variance_init=1e-3, 
                                 adaptation_rate=0.01, window_size=10)

# Butterworth Filter
# Note: The data length must be greater than 3*(filter order) + 1
# Higher order filters require more data points
# Use lower order for small datasets
bf_low = prices.butterworth_filter(cutoff=0.1, order=2, filter_type='lowpass')  # Order 2 requires at least 7 points

# For higher order filters, use longer data
# Order 4 requires at least 13 data points, order 6 requires 19 points, etc.
bf_high = longer_prices.butterworth_filter(cutoff=0.1, order=4, filter_type='highpass')
bf_band = longer_prices.butterworth_filter(cutoff=(0.1, 0.4), order=4, filter_type='bandpass')

# Savitzky-Golay Filter
sg = prices.savitzky_golay_filter(window_length=11, polyorder=3)

# Hampel Filter (for outlier removal)
hf = prices.hampel_filter(window_size=5, n_sigmas=3.0)

# Hodrick-Prescott Filter
trend, cycle = prices.hodrick_prescott_filter(lambda_param=1600.0)
```

## Data Transformations

```python
# Normalize
normalized = prices.normalize(method='l2')
l1_norm = prices.normalize_l1()
l2_norm = prices.normalize_l2()

# Standardize
standardized = prices.standardize()

# Min-Max Scale
minmax = prices.min_max_scale(feature_range=(0, 1))

# Robust Scale
robust = prices.robust_scale(method='iqr', quantile_range=(25.0, 75.0))

# Quantile Transform
quantile = prices.quantile_transform(n_quantiles=1000, output_distribution='uniform')

# Winsorize
winsorized = prices.winsorize(limits=0.05)

# Remove Outliers
clean = prices.remove_outliers(method='zscore', threshold=3.0)

# Fill Missing Values
filled = prices.fill_missing(method='mean')

# Interpolate Missing Values
interpolated = prices.interpolate_missing(method='linear')

# Log Transform
# Note: Use offset for zero/negative values
log_data = prices.log_transform(base=None, offset=1.0 if prices.min() <= 0 else 0.0)

# Power Transform
power_transformed = prices.power_transform(method='yeo-johnson')

# Scale to Range
scaled = prices.scale_to_range(feature_range=(0.0, 1.0))

# Clip Outliers
clipped = prices.clip_outliers(lower_percentile=1.0, upper_percentile=99.0)

# Discretize
discretized = prices.discretize(n_bins=5, strategy='uniform')

# Polynomial Features
poly = prices.polynomial_features(degree=2)

# Resample
resampled = prices.resample(factor=2, method='mean')

# Dynamic Tanh
tanh_transformed = prices.dynamic_tanh(alpha=1.0)
```

## Time Series Operations

```python
# Create Lag Features
lagged = prices.lag_features(lags=[1, 2, 3])

# Calculate Slope
slope = prices.slope(period=5)

# Wave Function
wave = prices.wave(high=high, low=low, close=prices)

# ZigZag
zigzag = prices.zigzag(threshold=0.03)

# Rolling Window Statistics
roll_stats = prices.rolling_statistics(window=10, statistics=['mean', 'std', 'skew', 'kurt'])
roll_mean = roll_stats['mean']
roll_std = roll_stats['std']
```

## Advanced Usage

### Method Chaining

One of the most powerful features of Parray is the ability to chain methods:

```python
# Complex analysis in a single chain
result = (
    prices
    .remove_outliers(method='zscore', threshold=3.0)
    .fill_missing(method='mean')
    .log_transform(offset=1.0 if prices.min() <= 0 else 0.0)
    .ema(period=5)
    .standardize()
)

# Creating a custom indicator
custom_indicator = (
    (prices.ema(5) - prices.ema(20)) /  # Fast MA - Slow MA
    prices.atr(high, low, 14)            # Normalized by ATR
)
```

### Custom Functions with apply and apply_along_axis

```python
# Apply a custom function to each element
# Note: When using apply(), the function must handle entire arrays at once
def custom_func_array(x):
    # Use vectorized operations instead of if-else
    result = np.empty_like(x, dtype=float)
    positive_mask = x > 0
    result[positive_mask] = np.sin(x[positive_mask])
    result[~positive_mask] = np.cos(x[~positive_mask])
    return result

result = prices.apply(custom_func_array)

# Alternative: Use numpy's vectorize to convert element-wise function to array function
@np.vectorize
def custom_func_element(x):
    # This operates on single elements
    return np.sin(x) if x > 0 else np.cos(x)

result = prices.apply(lambda x: custom_func_element(x))

# Apply a function along an axis (for 2D arrays)
def row_func(row):
    return np.sum(row) / np.max(row)

result_2d = multi_series.apply_along_axis(row_func, axis=1)

# Rolling apply
def roll_func(window):
    return np.max(window) - np.min(window)

roll_range = prices.rolling_apply(window=5, func=roll_func)
```

## Best Practices

### Performance Optimization

1. **Enable GPU for large datasets**:
   ```python
   if Parray.is_gpu_available():
       prices.enable_gpu()
   ```

2. **Use parallel processing for CPU-bound operations**:
   ```python
   prices.enable_parallel(num_workers=4)
   ```

3. **Process very large datasets in chunks**:
   ```python
   chunks = prices.to_chunks(chunk_size=100000)
   results = [process_chunk(chunk) for chunk in chunks]
   final = Parray.from_chunks(results)
   ```

4. **Use memory optimization for memory-intensive operations**:
   ```python
   prices.optimize_memory()
   ```

### Technical Analysis

1. **Combine multiple indicators for confirmation**:
   ```python
   buy_signal = (
       prices.crossover(prices.sma(20)) &  # Price crosses above MA
       (prices.rsi(14) < 70) &            # RSI not overbought
       (prices.adx(14) > 25)              # Strong trend
   )
   ```

2. **Use proper normalization for comparing instruments**:
   ```python
   # Compare two instruments on the same scale
   stock1_norm = stock1_prices.standardize()
   stock2_norm = stock2_prices.standardize()
   ```

3. **Consider time frame alignment**:
   ```python
   # Daily chart
   daily_signal = daily_prices.crossover(daily_prices.sma(200))
   
   # Weekly chart confirmation
   weekly_signal = weekly_prices.rsi(14) < 50
   ```

4. **Maintain good data hygiene**:
   ```python
   # Clean data before analysis
   clean_prices = (
       prices
       .remove_outliers()
       .fill_missing(method='forward')
       .interpolate_missing(method='linear')
   )
   ```


## Conclusion

The `Parray` class offers a powerful, flexible approach to financial time series analysis with NumPy's performance and a rich set of financial indicators and statistical tools. By leveraging method chaining, GPU acceleration, and parallel processing, you can perform complex analyses with clear, concise code. 