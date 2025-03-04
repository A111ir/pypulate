---
title: Technical Analysis
---

# Technical Analysis

The `pypulate.technical` module provides a comprehensive set of technical analysis tools for financial time series data. This guide demonstrates practical usage with real market data.

## Quick Start

Let's start with some sample market data:

```python
import numpy as np
from pypulate import Parray

# Sample market data
np.random.seed(42)
days = 200
price = np.cumsum(np.random.normal(0, 1, days)) + 100

# Convert to Parray for analysis
close = Parray(price)
```

## Momentum Indicators

### Relative Strength Index (RSI)

RSI measures momentum on a scale of 0 to 100, with readings above 70 indicating overbought conditions and below 30 indicating oversold conditions.

```python
# Calculate RSI with 14-period lookback
rsi = close.rsi(14)
print(f"Latest RSI: {rsi[-1]:.2f}")
```

### Moving Average Convergence Divergence (MACD)

MACD shows the relationship between two moving averages of a price series. Note that MACD calculation requires enough data points to compute both moving averages - the minimum required length is the slow period (typically 26 points).

```python
# Calculate MACD (12, 26, 9)
macd_line, signal_line, histogram = close.macd(12, 26, 9)
print(f"MACD Line: {macd_line[-1]:.2f}")
print(f"Signal Line: {signal_line[-1]:.2f}")
print(f"Histogram: {histogram[-1]:.2f}")
```

Note: The traditional MACD settings (12, 26, 9) require at least 26 data points. For shorter time series:
- Consider using shorter periods
- Ensure your data length is sufficient for the chosen periods
- The minimum data length needed = slow_period (second parameter)

## Volatility Indicators

### Bollinger Bands

Bollinger Bands consist of a middle band (20-day SMA) with upper and lower bands 2 standard deviations away.

```python
# Calculate Bollinger Bands
upper_bb, middle_bb, lower_bb = close.bollinger_bands(20, 2.0)
print(f"Upper Band: {upper_bb[-1]:.2f}")
print(f"Middle Band: {middle_bb[-1]:.2f}")
print(f"Lower Band: {lower_bb[-1]:.2f}")
```

## Trend Indicators

### Moving Averages

```python
# Calculate different types of moving averages
sma_20 = close.sma(20)  # Simple Moving Average
ema_20 = close.ema(20)  # Exponential Moving Average
wma_20 = close.wma(20)  # Weighted Moving Average

print(f"20-day SMA: {sma_20[-1]:.2f}")
print(f"20-day EMA: {ema_20[-1]:.2f}")
print(f"20-day WMA: {wma_20[-1]:.2f}")
```

## Building Trading Strategies

Here's an example of combining multiple indicators for a trading strategy:

```python
# Calculate indicators
rsi = close.rsi(14)
macd_line, signal, hist = close.macd(12, 26, 9)
upper_bb, middle_bb, lower_bb = close.bollinger_bands(20, 2.0)

# Generate trading signals
buy_signals = (
    (rsi < 30) &                  # RSI oversold
    (macd_line > signal) &        # MACD bullish
    (close < lower_bb)            # Price below lower Bollinger Band
)

sell_signals = (
    (rsi > 70) &                  # RSI overbought
    (macd_line < signal) &        # MACD bearish
    (close > upper_bb)            # Price above upper Bollinger Band
)

# Print latest signals
print("Latest Signals:")
print(f"Buy Signal: {buy_signals[-1]}")
print(f"Sell Signal: {sell_signals[-1]}")
```

## Advanced Analysis

### Logarithmic Returns

Calculate and analyze logarithmic returns:

```python
# Calculate log returns
log_returns = close.log().diff()
print(f"Latest Log Return: {log_returns[-1]:.4f}")

# Calculate RSI on log returns
log_rsi = close.log().rsi(14)
print(f"RSI of Log Returns: {log_rsi[-1]:.2f}")
```

### Statistical Measures

```python
# Calculate rolling statistics
volatility = close.rolling_std(20)
zscore = close.zscore(20)

print(f"20-day Volatility: {volatility[-1]:.2f}")
print(f"20-day Z-Score: {zscore[-1]:.2f}")
```

## Utility Functions

The module provides various utility functions for common calculations:

```python
from pypulate.technical.utils import rolling_max, rolling_min, slope

# Calculate 20-day high and low
high_20 = rolling_max(close, 20)
low_20 = rolling_min(close, 20)

# Calculate price slope
price_slope = slope(close, 5)

print(f"20-day High: {high_20[-1]:.2f}")
print(f"20-day Low: {low_20[-1]:.2f}")
print(f"5-day Slope: {price_slope[-1]:.4f}")

# Calculate top and high from slope of moving averages
top_low = close.sma(20).slope()
```
