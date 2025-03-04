---
title: Transforms
---

# Transforms

Pypulate provides transforms for identifying patterns especialy price action patterns in financial time series data. This page explains the available transforms and how to use them.

## Overview

Transforms in Pypulate are functions that convert price data into a different representation to identify patterns or significant points. Currently, Pypulate supports two main transforms:

1. **Wave Transform**: Identifies wave patterns in price data
2. **ZigZag Transform**: Identifies significant highs and lows

## Wave Transform

The wave transform converts OHLC data into a line without losing highs and lows, inspired by Glenn Neely wave chart.

### Parameters

- `open`: Open prices array
- `high`: High prices array
- `low`: Low prices array
- `close`: Close prices array

### Usage

```python
from pypulate.transforms import wave
from pypulate import Parray

# Real gold (XAU) OHLC data sample
data = {
    'open': [1936.13, 1935.33, 1938.06, 1947.38, 1943.64, 1942.30, 1947.15, 1945.40, 1944.72, 1943.69,
             1940.41, 1939.15, 1942.55, 1939.68, 1944.19, 1943.61, 1941.12, 1939.94, 1942.98, 1944.50],
    'high': [1937.48, 1938.79, 1948.68, 1949.05, 1944.51, 1947.70, 1947.71, 1946.24, 1947.87, 1945.06,
             1942.03, 1944.03, 1942.61, 1944.45, 1952.94, 1943.61, 1941.34, 1944.02, 1946.06, 1946.32],
    'low': [1935.16, 1934.91, 1936.62, 1943.12, 1942.04, 1941.94, 1944.43, 1943.19, 1940.27, 1939.03,
            1939.34, 1938.66, 1938.17, 1938.40, 1940.06, 1934.44, 1939.48, 1939.35, 1942.94, 1940.76],
    'close': [1935.33, 1938.09, 1947.36, 1943.64, 1942.35, 1947.14, 1945.40, 1944.72, 1943.70, 1940.43,
              1940.04, 1942.55, 1939.70, 1944.20, 1943.68, 1941.12, 1940.10, 1942.96, 1944.50, 1940.95]
}

waves = wave(data['open'], data['high'], data['low'], data['close'])
```

### Interpreting Results

The wave transform returns a 1D array containing the wave points.

### Example

```python
import matplotlib.pylab as plt
import numpy as np

# Plot the results
plt.figure(figsize=(12, 6))

for i in range(len(data['close'])):
    plt.plot([i, i], [data['low'][i], data['high'][i]], 'k-', alpha=0.3)
    
    if data['close'][i] >= data['open'][i]:
        body_color = 'green'
    else:
        body_color = 'red'
    
    plt.plot([i, i], [data['open'][i], data['close'][i]], color=body_color, linewidth=4)

if len(waves) > 0: 
    indices = np.arange(len(waves))
    plt.plot(indices, waves, 'b-o', linewidth=2, markersize=5, label='Wave Points')

plt.title('Wave Pattern Detection with Gold (XAU) OHLC Data')
plt.xlabel('Time Period')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## ZigZag Transform

The ZigZag transform identifies significant highs and lows in price data by filtering out smaller price movements.

### Parameters

- `threshold`: Minimum percentage change (0.03 = 3%) required to identify a new pivot point (default: 0.03)

### Usage

```python
import numpy as np
from pypulate.transforms import zigzag
from pypulate import Parray

# Real gold (XAU) price data
price = [1935.33, 1938.09, 1947.36, 1943.64, 1942.35, 1947.14, 1945.40, 1944.72, 1943.70, 1940.43,
         1940.04, 1942.55, 1939.70, 1944.20, 1943.68, 1941.12, 1940.10, 1942.96, 1944.50, 1940.95]
p_array = Parray(price)

# Method 1: Using the function directly
zz = zigzag(price, threshold=0.0005)  # 0.05% threshold for gold which is less volatile

# Method 2: Using Parray method chaining
zz = p_array.zigzag(threshold=0.0005)  # 0.05% threshold
```

### Interpreting Results

The zigzag transform returns a array with zigzag pivot points.

### Example

```python
import matplotlib.pylab as plt

# Plot zigzag points and lines
plt.figure(figsize=(12, 6))
plt.plot(price, label='Gold Price', alpha=0.7, color='gold')

if zz.size > 0:
    plt.plot(zz[:, 0], zz[:, 1], 'ro-', linewidth=2, label='ZigZag')

plt.title('ZigZag Pattern Detection on Gold (XAU) Prices')
plt.xlabel('Time Period')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Combining Wave and ZigZag Transforms

You can control the wave transform with zigzag as it will filter the mini changes in direction of line by zigzag threshold. This combination is particularly useful for identifying significant wave patterns while filtering out market noise.

### Example: Filtered Wave Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from pypulate import Parray
from pypulate.transforms import wave, zigzag

# Sample OHLC data
data = {
    'open': [1936.13, 1935.33, 1938.06, 1947.38, 1943.64, 1942.30, 1947.15, 1945.40, 1944.72, 1943.69,
             1940.41, 1939.15, 1942.55, 1939.68, 1944.19, 1943.61, 1941.12, 1939.94, 1942.98, 1944.50],
    'high': [1937.48, 1938.79, 1948.68, 1949.05, 1944.51, 1947.70, 1947.71, 1946.24, 1947.87, 1945.06,
             1942.03, 1944.03, 1942.61, 1944.45, 1952.94, 1943.61, 1941.34, 1944.02, 1946.06, 1946.32],
    'low': [1935.16, 1934.91, 1936.62, 1943.12, 1942.04, 1941.94, 1944.43, 1943.19, 1940.27, 1939.03,
            1939.34, 1938.66, 1938.17, 1938.40, 1940.06, 1934.44, 1939.48, 1939.35, 1942.94, 1940.76],
    'close': [1935.33, 1938.09, 1947.36, 1943.64, 1942.35, 1947.14, 1945.40, 1944.72, 1943.70, 1940.43,
              1940.04, 1942.55, 1939.70, 1944.20, 1943.68, 1941.12, 1940.10, 1942.96, 1944.50, 1940.95]
}

# Convert to numpy arrays
open_prices = np.array(data['open'])
high_prices = np.array(data['high'])
low_prices = np.array(data['low'])
close_prices = np.array(data['close'])

# Step 1: Calculate wave points
wave_points = wave(open_prices, high_prices, low_prices, close_prices)

# Step 2: Apply zigzag to filter the wave points
# Create a Parray from wave points to use the zigzag method
wave_parray = Parray(wave_points)
filtered_wave = wave_parray.zigzag(threshold=0.005)  # 0.5% threshold

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot OHLC data as candlesticks
for i in range(len(close_prices)):
    # Plot vertical line from low to high (wick)
    ax.plot([i, i], [low_prices[i], high_prices[i]], 'k-', alpha=0.3)
    
    # Determine candle color
    if close_prices[i] >= open_prices[i]:
        # Bullish candle (close > open)
        body_color = 'green'
    else:
        # Bearish candle (open > close)
        body_color = 'red'
    
    # Plot candle body
    ax.plot([i, i], [open_prices[i], close_prices[i]], color=body_color, linewidth=4)

# Plot original wave points
indices = np.arange(len(wave_points))
ax.plot(indices, wave_points, 'b-', linewidth=1, alpha=0.5, label='Wave Points')

# Plot filtered wave points
if filtered_wave.size > 0:
    ax.plot(filtered_wave[:, 0], filtered_wave[:, 1], 'ro-', linewidth=2, markersize=5, label='Filtered Wave Points')

# Add labels and styling
ax.set_title('Filtered Wave Pattern Detection with ZigZag')
ax.set_xlabel('Time Period')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

This technique provides several benefits:

1. **Noise Reduction**: The zigzag transform filters out minor price fluctuations in the wave pattern
2. **Trend Identification**: Helps identify the true underlying trend by focusing on significant price movements
3. **Signal Clarity**: Reduces false signals by eliminating small reversals that don't meet the threshold criteria
4. **Visualization Enhancement**: Creates a cleaner chart that highlights important price levels and potential reversal points
5. **Price Action**: This method is super useful for detecting price action patterns from price movement maybe better than candlesticks.

You can adjust the threshold parameter to control the sensitivity of the filtering. A higher threshold will result in fewer, more significant pivot points, while a lower threshold will capture more minor price movements. 

