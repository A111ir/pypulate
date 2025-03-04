---
title: Basic Usage
---

# Basic Usage

This guide demonstrates the basic usage of the `pypulate` package, focusing on the `Parray` class and various financial analysis methods.

## Creating a Parray

The `Parray` class is a wrapper around NumPy arrays that provides method chaining for financial analysis.

```python
import numpy as np
from pypulate import as_parray

# Create sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Convert to Parray
ts = as_parray(data)
```

## Moving Averages

```python
# Calculate Simple Moving Average
sma_values = ts.sma(period=3)

# Calculate Exponential Moving Average
ema_values = ts.ema(period=3)

# Chain multiple moving averages
result = ts.ema(3).sma(2)
```

## Momentum Indicators

```python
# Calculate RSI
rsi_values = ts.rsi(period=14)

# Calculate MACD
macd_line, signal_line, histogram = ts.macd()

# Calculate Stochastic Oscillator
k, d = ts.stochastic_oscillator()
```

## Volatility Measurements

```python
# Calculate Bollinger Bands
upper, middle, lower = ts.bollinger_bands(period=20, std_dev=2.0)

# Calculate ATR
atr_values = ts.atr(period=14)
```

## Transforms

```python
# Detect waves
waves = ts.wave(min_size=3)

# Apply ZigZag transform
zigzag_values = ts.zigzag(deviation=5.0, backstep=3)
```

## Filters

```python
# Apply Kalman filter
filtered_values = ts.kalman_filter()

# Apply Butterworth filter
filtered_values = ts.butterworth_filter(cutoff=0.1, order=4, filter_type='lowpass')
```

## Utility Functions

```python
# Calculate slope
slope_values = ts.slope(period=5)

# Calculate rolling statistics
max_values = ts.rolling_max(period=14)
min_values = ts.rolling_min(period=14)
std_values = ts.rolling_std(period=14)
```

## Method Chaining

One of the key features of `pypulate` is the ability to chain methods together for complex analysis:

```python
# Chain multiple operations
result = (
    ts
    .ema(5)                          # Apply 5-period EMA first
    .rsi(14)                         # Calculate RSI on the EMA
    .bollinger_bands(20, 2.0)        # Calculate Bollinger Bands on the RSI
)

# More complex example
macd_line, signal, hist = ts.macd()
stoch_k, stoch_d = ts.stochastic_oscillator()

# Generate signals
buy_signals = (
    (ts.rsi(14) < 30) &                  # RSI oversold
    (macd_line.crossover(signal)) &      # MACD bullish crossover
    (stoch_k.crossover(stoch_d))         # Stochastic bullish crossover
)
``` 