---
title: Financial Analysis
---

# Financial Analysis Examples

This guide demonstrates various financial analysis strategies using the `pypulate` package.

## Technical Analysis

### Moving Average Crossover Strategy

```python
import numpy as np
from pypulate import as_parray

# Create sample price data
prices = np.array([100, 102, 104, 103, 105, 107, 108, 107, 105, 104, 103, 105, 107, 109, 108])

# Convert to Parray
ts = as_parray(prices)

# Calculate fast and slow moving averages
fast_ma = ts.ema(5)
slow_ma = ts.ema(10)

# Generate crossover signals
buy_signals = fast_ma.crossover(slow_ma)
sell_signals = fast_ma.crossunder(slow_ma)

print("Buy signals:", buy_signals)
print("Sell signals:", sell_signals)
```

### RSI Overbought/Oversold Strategy

```python
# Calculate RSI
rsi_values = ts.rsi(14)

# Generate signals
buy_signals = rsi_values < 30  # Oversold
sell_signals = rsi_values > 70  # Overbought

print("Buy signals (oversold):", buy_signals)
print("Sell signals (overbought):", sell_signals)
```

### MACD Strategy

```python
# Calculate MACD
macd_line, signal_line, histogram = ts.macd()

# Generate signals
buy_signals = macd_line.crossover(signal_line)
sell_signals = macd_line.crossunder(signal_line)

print("Buy signals (MACD crossover):", buy_signals)
print("Sell signals (MACD crossunder):", sell_signals)
```

### Bollinger Bands Strategy

```python
# Calculate Bollinger Bands
upper, middle, lower = ts.bollinger_bands(20, 2.0)

# Generate signals
buy_signals = ts < lower  # Price below lower band
sell_signals = ts > upper  # Price above upper band

print("Buy signals (price below lower band):", buy_signals)
print("Sell signals (price above upper band):", sell_signals)
```

## Volatility Analysis

### ATR-Based Position Sizing

```python
# Calculate ATR
atr_values = ts.atr(14)

# Define risk parameters
account_size = 10000  # $10,000 account
risk_percentage = 0.02  # 2% risk per trade
risk_amount = account_size * risk_percentage  # $200 risk per trade

# Calculate position size based on ATR
current_price = prices[-1]
current_atr = atr_values[-1]
stop_loss_distance = 2 * current_atr  # 2 ATR units for stop loss

# Position size calculation
position_size = risk_amount / stop_loss_distance
position_value = position_size * current_price

print(f"Current price: ${current_price}")
print(f"Current ATR: ${current_atr}")
print(f"Stop loss distance: ${stop_loss_distance}")
print(f"Position size: {position_size} shares")
print(f"Position value: ${position_value}")
```

### Volatility Regime Detection

```python
# Calculate historical volatility
hist_vol = ts.historical_volatility(21, 252)

# Define volatility regimes
low_vol = hist_vol < 15  # Less than 15% annualized volatility
medium_vol = (hist_vol >= 15) & (hist_vol < 30)
high_vol = hist_vol >= 30  # More than 30% annualized volatility

print("Low volatility regime:", low_vol)
print("Medium volatility regime:", medium_vol)
print("High volatility regime:", high_vol)
```

## Multi-Indicator Strategy

```python
# Create a more complex strategy combining multiple indicators
complex_buy_signals = (
    (ts.rsi(14) < 30) &                          # RSI oversold
    (ts.macd()[0].crossover(ts.macd()[1])) &     # MACD bullish crossover
    (ts < ts.bollinger_bands(20, 2.0)[2])        # Price below lower Bollinger Band
)

complex_sell_signals = (
    (ts.rsi(14) > 70) &                          # RSI overbought
    (ts.macd()[0].crossunder(ts.macd()[1])) &    # MACD bearish crossunder
    (ts > ts.bollinger_bands(20, 2.0)[0])        # Price above upper Bollinger Band
)

print("Complex buy signals:", complex_buy_signals)
print("Complex sell signals:", complex_sell_signals)
```

## Backtesting (Simplified)

```python
# Simplified backtesting example
def simple_backtest(prices, buy_signals, sell_signals):
    cash = 10000  # Initial cash
    shares = 0    # Initial shares
    position_value = 0
    
    for i in range(len(prices)):
        if buy_signals[i] and cash > 0:
            # Buy as many shares as possible
            shares = cash / prices[i]
            cash = 0
            print(f"Day {i}: BUY at ${prices[i]}, Shares: {shares}")
        
        elif sell_signals[i] and shares > 0:
            # Sell all shares
            cash = shares * prices[i]
            shares = 0
            print(f"Day {i}: SELL at ${prices[i]}, Cash: ${cash}")
        
        # Calculate current position value
        position_value = cash + (shares * prices[i])
        
    return position_value

# Generate signals for backtest
sma_fast = ts.sma(5)
sma_slow = ts.sma(10)
buy_signals = sma_fast.crossover(sma_slow)
sell_signals = sma_fast.crossunder(sma_slow)

# Run backtest
final_value = simple_backtest(prices, buy_signals, sell_signals)
print(f"Final portfolio value: ${final_value}")
``` 