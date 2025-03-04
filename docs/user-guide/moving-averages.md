# Moving Averages

Pypulate provides a comprehensive set of moving average functions for financial time series analysis. This 
page explains the different types of moving averages available and how to use them.

## Available Moving Averages

### Specialized Moving Averages

These moving averages are designed for specific use cases:

- **Volume-Weighted Moving Average (VWMA)**: Weights price by volume.
- **Kaufman Adaptive Moving Average (KAMA)**: Adapts to market volatility.
- **Arnaud Legoux Moving Average (ALMA)**: Reduces lag and noise.
- **Fractal Adaptive Moving Average (FRAMA)**: Adapts to market fractal dimension.
- **Jurik Moving Average (JMA)**: Reduces noise and lag.
- **Laguerre Filter**: Uses Laguerre polynomials for smoothing.
- **Least Squares Moving Average (LSMA)**: Uses linear regression.
- **McGinley Dynamic Indicator**: Adapts to market speed.
- **Modular Filter**: Adjusts smoothing based on phase.
- **Rex Dog Moving Average (RDMA)**: Average of six SMAs with different periods.
- **Tillson T3**: Triple EMA with reduced lag.
- **Volatility-Adjusted Moving Average (VAMA)**: Adjusts based on volatility.

## Using Moving Averages

### Functional Approach

You can use moving averages directly by importing the functions:

```python
import numpy as np
from pypulate.moving_averages import sma, ema, hma

# Create sample data
data = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]

# Calculate moving averages
sma_result = sma(data, period=3)
ema_result = ema(data, period=3)
hma_result = hma(data, period=3)
```

### Method Chaining with Parray

For a more fluent interface, you can use the `Parray` class:

```python
from pypulate import Parray

# Create sample data
data = Parray([10, 11, 12, 11, 10, 9, 10, 11, 12, 13])

# Calculate moving averages using method chaining
sma_result = Parray.sma(period=3)
ema_result = Parray.ema(period=3)
hma_result = Parray.hma(period=3)
```

## Examples

### Comparing Different Moving Averages

```python
import numpy as np
import matplotlib.pyplot as plt
from pypulate import Parray

# Generate sample price data
np.random.seed(42)
days = 100
price = np.cumsum(np.random.normal(0, 1, days)) + 100

# Convert to Parray for method chaining
price_array = Parray(price)

# Calculate different types of moving averages
sma = price_array.sma(20)
ema = price_array.ema(20)
wma = price_array.wma(20)
hma = price_array.hma(20)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(price, label='Price', alpha=0.5, color='gray')
plt.plot(sma, label='SMA(20)')
plt.plot(ema, label='EMA(20)')
plt.plot(wma, label='WMA(20)')
plt.plot(hma, label='HMA(20)')

plt.title('Comparison of Different Moving Averages')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Moving Average Crossover Strategy

```python
import numpy as np
import matplotlib.pyplot as plt
from pypulate import Parray

# Generate sample price data
np.random.seed(42)
days = 200
price = np.cumsum(np.random.normal(0, 1, days)) + 100

# Convert to Parray for method chaining
price_array = Parray(price)

# Calculate fast and slow EMAs
fast_ema = price_array.ema(9)
slow_ema = price_array.ema(21)

# Generate buy/sell signals
buy_signals = fast_ema.crossover(slow_ema)
sell_signals = fast_ema.crossunder(slow_ema)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(price, label='Price')
plt.plot(fast_ema, label='9-day EMA', alpha=0.7)
plt.plot(slow_ema, label='21-day EMA', alpha=0.7)

# Plot buy signals
buy_indices = np.where(buy_signals)[0]
plt.scatter(buy_indices, price[buy_indices], marker='^', color='green', s=100, label='Buy Signal')

# Plot sell signals
sell_indices = np.where(sell_signals)[0]
plt.scatter(sell_indices, price[sell_indices], marker='v', color='red', s=100, label='Sell Signal')

plt.title('Moving Average Crossover Strategy')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Choosing the Right Moving Average

Different moving averages are suitable for different market conditions:

- **Trending Markets**: SMA, EMA, WMA, HMA
- **Volatile Markets**: KAMA, ALMA, FRAMA, JMA
- **Ranging Markets**: DEMA, TEMA, T3

Experiment with different types to find the one that works best for your specific use case. 