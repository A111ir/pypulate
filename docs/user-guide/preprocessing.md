# Data Preprocessing

The preprocessing module provides a comprehensive set of tools for data preprocessing and statistical analysis, designed specifically for financial data.

## Getting Started with Parray

First, let's create a Parray object from your data:

```python
from pypulate.dtypes import Parray

# Create a Parray from numpy array
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ts = Parray(data)

# Enable GPU acceleration (if available)
ts.enable_gpu()

# Enable parallel processing
ts.enable_parallel(num_workers=4)
```

## Data Normalization and Scaling

### Normalize
```python
# L2 normalization (default)
normalized_data = ts.normalize(method='l2')

# L1 normalization
normalized_data = ts.normalize(method='l1')

# Using method chaining
result = ts.normalize_l2().standardize()
```

### Standardize
```python
# Z-score normalization
standardized_data = ts.standardize()

# With parallel processing
ts.enable_parallel()
standardized_data = ts.standardize()
```

### Min-Max Scaling
```python
# Scale to [0,1] range (default)
scaled_data = ts.min_max_scale()

# Scale to custom range
scaled_data = ts.min_max_scale(feature_range=(-1, 1))
```

### Robust Scaling
```python
# IQR-based scaling (default)
robust_data = ts.robust_scale(method='iqr')

# Custom quantile range
robust_data = ts.robust_scale(method='iqr', quantile_range=(10.0, 90.0))

# Using MAD scaling
robust_data = ts.robust_scale(method='mad')
```

## Outlier Handling

### Remove Outliers
```python
# Z-score based removal
cleaned_data = ts.remove_outliers(method='zscore', threshold=2.0)

# IQR based removal
cleaned_data = ts.remove_outliers(method='iqr')

# Using method chaining
result = ts.remove_outliers().standardize()
```

### Winsorize
```python
# Symmetric winsorization at 5% (default)
winsorized_data = ts.winsorize(limits=0.05)

# Asymmetric winsorization
winsorized_data = ts.winsorize(limits=(0.01, 0.05))
```

## Missing Value Handling

### Fill Missing Values
```python
# Mean imputation
filled_data = ts.fill_missing(method='mean')

# Median imputation
filled_data = ts.fill_missing(method='median')

# Custom value
filled_data = ts.fill_missing(method='constant', value=0.0)

# Forward fill
filled_data = ts.fill_missing(method='forward')

# Backward fill
filled_data = ts.fill_missing(method='backward')
```

### Interpolate Missing Values
```python
# Linear interpolation
interpolated_data = ts.interpolate_missing(method='linear')

# Cubic interpolation
interpolated_data = ts.interpolate_missing(method='cubic')

# Quadratic interpolation
interpolated_data = ts.interpolate_missing(method='quadratic')
```

## Time Series Operations

### Resampling
```python
# Downsample by factor of 2 using mean
resampled_data = ts.resample(factor=2, method='mean')

# Downsample using median
resampled_data = ts.resample(factor=2, method='median')

# Downsample using sum
resampled_data = ts.resample(factor=2, method='sum')
```

### Rolling Window Operations
```python
# Create rolling windows of size 5
windows = ts.rolling_window(window_size=5)

# Create rolling windows with step size 2
windows = ts.rolling_window(window_size=5, step=2)
```

### Lag Features
```python
# Create lag features for lags 1, 2, and 3
lagged_data = ts.lag_features(lags=[1, 2, 3])
```

## Transformations

### Log Transform
```python
# Natural log transform (only works with positive data)
log_data = ts.log_transform()

# Log transform with custom base
log_data = ts.log_transform(base=10)

# Log transform with offset for data containing zeros or negative values
# Add an offset to make all values positive before taking log
log_data = ts.log_transform(offset=1.0)  # For data with zeros
log_data = ts.log_transform(offset=abs(ts.min()) + 1)  # For data with negative values
```

### Power Transform
```python
# Yeo-Johnson transform
transformed_data = ts.power_transform(method='yeo-johnson')

# Box-Cox transform
transformed_data = ts.power_transform(method='box-cox')

# Without standardization
transformed_data = ts.power_transform(standardize=False)
```

### Dynamic Tanh Transform
```python
# Apply Dynamic Tanh transformation
dyt_data = ts.dynamic_tanh(alpha=1.0)

# More aggressive normalization
dyt_data = ts.dynamic_tanh(alpha=2.0)
```

## Statistical Analysis

### Descriptive Statistics
```python
# Calculate basic statistics
stats = ts.descriptive_stats()

# Access specific statistics
mean = stats['mean']
std = stats['std']
skewness = stats['skewness']
kurtosis = stats['kurtosis']
```

### Correlation Analysis
```python
# Create a 2D array with multiple variables
from pypulate import Parray

# Create a 2D Parray with 3 variables (columns)
ps = parray([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Calculate Pearson correlation
corr_matrix = ps.correlation_matrix(method='pearson')
print(corr_matrix)  # Shows correlation between the 3 columns

# Calculate Spearman correlation
corr_matrix = ps.correlation_matrix(method='spearman')

# Calculate Kendall correlation
corr_matrix = ps.correlation_matrix(method='kendall')

# For a single time series, you need to create features first
ps_single = Parray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
lagged_ts = ps_single.lag_features(lags=[1, 2, 3])  # Creates a 2D array with original and lagged features
corr_matrix = lagged_ts.correlation_matrix(method='pearson')  # Correlation between original and lagged values
```

### Stationarity Tests
```python
import numpy as np
from pypulate import Parray

# Create a sample for stationarity tests
np.random.seed(42) # For reproducibility
ts = Parray(np.random.normal(0, 1, size=100))  # 100 observations

# Perform ADF test
test_stat, p_value = ts.augmented_dickey_fuller_test()

# Perform KPSS test
test_stat, p_value = ts.kpss_test()

# Perform variance ratio test
# Note: For each period k, you need at least 2k+1 observations
# Important: variance_ratio_test requires strictly positive values (like stock prices)

# Create data that simulates a price series (strictly positive)
price_data = np.cumprod(1 + np.random.normal(0.001, 0.01, size=100))  # Random walk with drift
ts_prices = Parray(price_data)

# Use smaller periods for smaller samples
results_small = ts_prices.variance_ratio_test(periods=[2, 4, 8])

# For larger periods, need larger samples
long_price_data = np.cumprod(1 + np.random.normal(0.001, 0.01, size=500))  # Random walk with drift
long_ts_prices = Parray(long_price_data)
results_large = long_ts_prices.variance_ratio_test(periods=[2, 4, 8, 16, 32])
```

### Rolling Statistics
```python
# Calculate multiple rolling statistics
stats = ts.rolling_statistics(window=20, statistics=['mean', 'std', 'skew', 'kurt'])

# Access specific rolling statistics
rolling_mean = stats['mean']
rolling_std = stats['std']
```

## Performance Optimization

### GPU Acceleration
```python
# Enable GPU acceleration
ts.enable_gpu()

# Check GPU availability
if Parray.is_gpu_available():
    gpu_info = Parray.get_gpu_info()
    print(f"Using GPU: {gpu_info['devices'][0]['name']}")
```

### Parallel Processing
```python
# Enable parallel processing
ts.enable_parallel(num_workers=4)

# Custom chunk size
ts.enable_parallel(num_workers=4, chunk_size=10000)
```

### Memory Optimization
```python
# Optimize memory usage
ts.optimize_memory()

# Process large datasets in chunks
chunks = ts.to_chunks(chunk_size=10000)
results = [chunk.process() for chunk in chunks]
final_result = Parray.from_chunks(results)
```

## Best Practices

1. **Data Validation**: Always check your data for missing values and outliers before applying transformations.

2. **Scaling Order**: When applying multiple transformations, consider the order:
   - Handle missing values first
   - Remove outliers
   - Apply transformations (log, power)
   - Scale the data

3. **Time Series Considerations**: For time series data:
   - Check for stationarity
   - Consider using appropriate lag features
   - Be careful with interpolation methods

4. **Memory Efficiency**: The module is designed to be memory efficient, but for large datasets:
   - Use appropriate window sizes
   - Consider processing in chunks
   - Monitor memory usage with large transformations

5. **Performance Optimization**:
   - Enable GPU acceleration when available
   - Use parallel processing for large datasets
   - Optimize memory usage when working with large arrays
   - Consider using method chaining for better readability

## Performance Tips

1. **Vectorized Operations**: All functions are vectorized for optimal performance.

2. **Memory Management**: Functions return NumPy arrays, which are memory efficient.

3. **Missing Value Handling**: Functions handle NaN values appropriately without raising errors.

4. **Type Safety**: Functions use type hints and input validation for reliability.

5. **GPU Acceleration**: Enable GPU acceleration for supported operations to improve performance.

6. **Parallel Processing**: Use parallel processing for large datasets and computationally intensive operations.

7. **Method Chaining**: Take advantage of method chaining for cleaner and more efficient code:

```python
# Example of method chaining
# Note: log_transform requires positive values, so we add offset if needed
result = (ts
    .remove_outliers()
    .fill_missing(method='mean')
    .standardize()
    # Add sufficient offset for log transform if data might contain zero/negative values
    .log_transform(offset=abs(ts.min()) + 1 if ts.min() <= 0 else 0)
    .rolling_statistics(window=20, statistics=['mean', 'std'])
)
``` 