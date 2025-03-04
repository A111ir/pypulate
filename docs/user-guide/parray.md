# Using the Parray Class

The `Parray` class is a powerful tool for financial time series analysis in Pypulate. It extends NumPy arrays with financial analysis methods that can be chained together for complex calculations.

## Introduction

`Parray` is designed to make financial time series analysis more intuitive and concise. It inherits from `numpy.ndarray`, so it has all the functionality of NumPy arrays, plus additional methods for financial analysis.

## Getting Started

### Creating a Parray

You can create a `Parray` object from any array-like object using the `Parray` class:

```python
from pypulate.dtypes import Parray

# From a list
data = [1, 2, 3, 4, 5]
p = Parray(data)

# From a NumPy array
data = np.array([1, 2, 3, 4, 5])
p = Parray(data)
```


## Moving Averages

`Parray` provides methods for calculating various moving averages:

```python
from pypulate import Parray

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
p = Parray(data)

# Simple Moving Average
sma = p.sma(3)  # 3-period SMA

# Exponential Moving Average
ema = p.ema(3)  # 3-period EMA

# Weighted Moving Average
wma = p.wma(3)  # 3-period WMA

# Hull Moving Average
hma = p.hma(3)  # 3-period HMA
```

## Crossover and Crossunder Detection

One of the most useful features of `Parray` is the ability to detect crossovers and crossunders, which are common signals in technical analysis:

```python
from pypulate.dtypes import Parray

# Create sample price data
data = Parray([10, 11, 12, 11, 10, 11, 12, 13])

# Find where prices cross above 11
crossover_points = data.crossover(11)

# Find where prices cross below 11
crossunder_points = data.crossunder(11)

# Find where prices cross above a moving average
ma = data.sma(3)
crossover_ma = data.crossover(ma)
```

## Advanced Technical Analysis

### Oscillators and Momentum

```python
from pypulate import Parray

# Sample price data
prices = Parray([
    100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
    111, 113, 112, 115, 114, 116, 118, 117, 119, 120
])

# RSI (Relative Strength Index)
rsi = prices.rsi(14)  # 14-period RSI


# MACD (Moving Average Convergence Divergence)
macd_line, signal_line, histogram = prices.macd(6, 12, 6)

# Rate of Change (ROC)
roc = prices.roc(12)  # 12-period rate of change

# Momentum
momentum = prices.momentum(10)  # 10-period momentum

```

### Volatility Indicators

```python
# Sample OHLC data
high = Parray([
    102, 104, 105, 104, 107, 108, 107, 109, 111, 110,
    112, 114, 113, 116, 115, 117, 119, 118, 120, 121
])
low = Parray([
    99, 101, 103, 102, 104, 106, 105, 107, 109, 108,
    110, 112, 111, 114, 113, 115, 117, 116, 118, 119
])
close = Parray([
    100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
    111, 113, 112, 115, 114, 116, 118, 117, 119, 120
])

# Bollinger Bands (uses close prices)
upper_band, middle_band, lower_band = close.bollinger_bands(12, 2)  # 12-period, 2 standard deviations

# Average True Range (ATR) - requires high, low, close prices
atr = Parray.atr(high, low, close, 14)  # 14-period ATR

# Keltner Channels (uses high, low, close for ATR calculation)
k_upper, k_middle, k_lower = Parray.keltner_channels(close, high, low, 12, 2)  # 12-period, 2 ATR multiplier

# Standard Deviation
std = close.rolling_std(12)  # 12-period rolling standard deviation

# CCI with typical price
cci = close.typical_price(high, low).cci(9)
```

### Trade Strategy

```python
# Calculate indicators
adx = close.adx(14)  # 14-period ADX
ma_10 = close.sma(10)  # 10-period SMA

# Generate long position signals
# Long when: price > MA(10) AND ADX < 30 (weak trend, good for entry)
long_signals = (close > ma_10) & (adx < 30)

# Get indices where long_signals is True
print("Long Entry Points:", np.where(long_signals)[0])

```


### Custom Indicators

```python
# Create custom indicators using method chaining
custom_ma = (
    prices.sma(10) * 0.5 +      # 50% weight to SMA
    prices.ema(10) * 0.3 +      # 30% weight to EMA
    prices.wma(10) * 0.2        # 20% weight to WMA
)

# Custom momentum indicator
custom_momentum = (
    prices.roc(10) * 0.4 +      # 40% weight to ROC
    prices.rsi(14) * 0.3 +      # 30% weight to RSI
    prices.momentum(10) * 0.3   # 30% weight to Momentum
)
```

## Best Practices

1. **Signal Generation**:
   - Combine multiple indicators for confirmation
   - Use proper risk management
   - Consider time frame alignment
   - Filter out noise with minimum threshold values

## Performance Considerations

Since `Parray` is a subclass of `numpy.ndarray`, it inherits all of NumPy's performance characteristics. However, there are a few things to keep in mind:

1. Method chaining creates intermediate arrays, which can increase memory usage for very large datasets.
2. For extremely performance-critical applications, you may want to use the underlying functions directly.

For most use cases, the convenience of method chaining outweighs any minor performance impact. 