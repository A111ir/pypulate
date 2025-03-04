---
title: Advanced Techniques
---

# Advanced Techniques

This guide demonstrates advanced techniques and methods using the `pypulate` package.

## Advanced Method Chaining

### Multi-Indicator Strategy with Method Chaining

```python
import numpy as np
from pypulate import as_parray

# Create sample price data
prices = np.array([100, 102, 104, 103, 105, 107, 108, 107, 105, 104, 103, 105, 107, 109, 108])
high = prices + 2
low = prices - 2
volume = np.array([1000, 1200, 1500, 1300, 1400, 1600, 1700, 1500, 1300, 1200, 1100, 1300, 1500, 1700, 1600])

# Convert to Parray
ts = as_parray(prices)

# Create a complex strategy with method chaining
# 1. Calculate price-based indicators
ema_short = ts.ema(5)
ema_long = ts.ema(10)
upper_bb, middle_bb, lower_bb = ts.bollinger_bands(20, 2.0)

# 2. Calculate momentum indicators
rsi_values = ts.rsi(14)
macd_line, signal_line, histogram = ts.macd()
k, d = ts.stochastic_oscillator(high, low)

# 3. Calculate volatility indicators
atr_values = ts.atr(high, low, 14)
volatility = ts.historical_volatility(21)

# 4. Generate complex signals
buy_signals = (
    (ema_short.crossover(ema_long)) &                # Price trend confirmation
    (rsi_values > 30) & (rsi_values < 70) &          # RSI not extreme
    (macd_line.crossover(signal_line)) &             # MACD bullish crossover
    (k.crossover(d)) & (k < 50) &                    # Stochastic bullish crossover from low levels
    (ts > lower_bb) & (ts < middle_bb) &             # Price between lower and middle BB
    (volatility < 30)                                # Low volatility environment
)

sell_signals = (
    (ema_short.crossunder(ema_long)) &               # Price trend confirmation
    (rsi_values > 70) &                              # RSI overbought
    (macd_line.crossunder(signal_line)) &            # MACD bearish crossunder
    (k.crossunder(d)) & (k > 50) &                   # Stochastic bearish crossunder from high levels
    (ts > middle_bb) & (ts < upper_bb)               # Price between middle and upper BB
)

print("Complex buy signals:", buy_signals)
print("Complex sell signals:", sell_signals)
```

## Custom Indicators

### Creating a Custom Indicator

```python
import numpy as np
from pypulate import as_parray

# Create a custom indicator function
def custom_momentum_oscillator(data, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, weight_rsi=0.5):
    """
    Custom momentum oscillator combining RSI and MACD
    
    Parameters
    ----------
    data : numpy.ndarray
        Input price data
    rsi_period : int
        Period for RSI calculation
    macd_fast : int
        Fast period for MACD
    macd_slow : int
        Slow period for MACD
    macd_signal : int
        Signal period for MACD
    weight_rsi : float
        Weight for RSI component (0-1)
        
    Returns
    -------
    numpy.ndarray
        Custom momentum oscillator values
    """
    # Convert to Parray
    ts = as_parray(data)
    
    # Calculate RSI (0-100)
    rsi = ts.rsi(rsi_period)
    
    # Calculate MACD
    macd_line, signal_line, histogram = ts.macd(macd_fast, macd_slow, macd_signal)
    
    # Normalize MACD histogram to 0-100 scale
    # Find min and max values
    hist_min = np.nanmin(histogram)
    hist_max = np.nanmax(histogram)
    
    # Normalize to 0-100
    if hist_max - hist_min != 0:
        normalized_hist = ((histogram - hist_min) / (hist_max - hist_min)) * 100
    else:
        normalized_hist = np.full_like(histogram, 50)
    
    # Combine RSI and normalized MACD histogram
    weight_macd = 1 - weight_rsi
    custom_oscillator = (rsi * weight_rsi) + (normalized_hist * weight_macd)
    
    return custom_oscillator

# Apply the custom indicator
prices = np.array([100, 102, 104, 103, 105, 107, 108, 107, 105, 104, 103, 105, 107, 109, 108])
custom_indicator = custom_momentum_oscillator(prices)

print("Custom Momentum Oscillator:", custom_indicator)

# Generate signals
overbought = custom_indicator > 70
oversold = custom_indicator < 30

print("Overbought signals:", overbought)
print("Oversold signals:", oversold)
```

## Advanced Filtering Techniques

### Combining Multiple Filters

```python
import numpy as np
from pypulate import as_parray

# Create noisy price data
np.random.seed(42)
base_prices = np.linspace(100, 150, 100)  # Upward trend
noise = np.random.normal(0, 5, 100)       # Random noise
prices = base_prices + noise

# Convert to Parray
ts = as_parray(prices)

# Apply multiple filters in sequence
filtered_prices = (
    ts
    .kalman_filter(process_variance=1e-4, measurement_variance=1e-2)  # First pass with Kalman filter
    .butterworth_filter(cutoff=0.1, order=3, filter_type='lowpass')   # Remove high-frequency noise
    .savitzky_golay_filter(window_length=11, polyorder=3)             # Smooth the result
)

# Calculate the noise reduction
original_variance = np.var(prices)
filtered_variance = np.var(filtered_prices)
noise_reduction_percent = (1 - (filtered_variance / original_variance)) * 100

print(f"Original price variance: {original_variance:.2f}")
print(f"Filtered price variance: {filtered_variance:.2f}")
print(f"Noise reduction: {noise_reduction_percent:.2f}%")
```

### Adaptive Filtering Based on Volatility

```python
import numpy as np
from pypulate import as_parray

# Create price data with varying volatility
np.random.seed(42)
n_points = 200
base_prices = np.linspace(100, 200, n_points)  # Upward trend

# Create periods of different volatility
volatility = np.ones(n_points) * 2  # Base volatility
volatility[50:100] = 5              # Higher volatility in this region
volatility[150:] = 8                # Even higher volatility at the end

# Generate prices with varying volatility
noise = np.random.normal(0, volatility)
prices = base_prices + noise

# Convert to Parray
ts = as_parray(prices)

# Calculate historical volatility
hist_vol = ts.historical_volatility(21)

# Apply adaptive filtering based on volatility
result = np.full_like(prices, np.nan)

for i in range(len(prices)):
    if i < 21:  # Not enough data for volatility calculation
        result[i] = prices[i]
    else:
        current_vol = hist_vol[i]
        
        # Choose filter parameters based on volatility
        if current_vol < 10:  # Low volatility
            # Light filtering
            window_size = 3
            n_sigmas = 2.0
        elif current_vol < 20:  # Medium volatility
            # Medium filtering
            window_size = 5
            n_sigmas = 2.5
        else:  # High volatility
            # Strong filtering
            window_size = 7
            n_sigmas = 3.0
        
        # Apply Hampel filter with adaptive parameters
        if i >= window_size:
            window = prices[i-window_size:i+1]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            if mad == 0:  # Avoid division by zero
                result[i] = prices[i]
            else:
                # Scale MAD to estimate standard deviation
                sigma = 1.4826 * mad
                
                # Check if the point is an outlier
                if abs(prices[i] - median) > n_sigmas * sigma:
                    result[i] = median  # Replace outlier with median
                else:
                    result[i] = prices[i]  # Keep original value
        else:
            result[i] = prices[i]

# Convert result to Parray for further analysis
filtered_ts = as_parray(result)

print("Adaptive filtering complete")
```

## Advanced Time Series Decomposition

### Trend-Cycle Decomposition with Hodrick-Prescott Filter

```python
import numpy as np
import matplotlib.pyplot as plt
from pypulate import as_parray

# Create sample price data with trend, cycle, and noise
np.random.seed(42)
n_points = 200

# Create trend component
trend = np.linspace(100, 200, n_points)

# Create cyclical component (sine wave)
cycle_period = 40
cycle = 15 * np.sin(2 * np.pi * np.arange(n_points) / cycle_period)

# Create seasonal component (smaller sine wave)
seasonal_period = 10
seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / seasonal_period)

# Add random noise
noise = np.random.normal(0, 3, n_points)

# Combine components
prices = trend + cycle + seasonal + noise

# Convert to Parray
ts = as_parray(prices)

# Apply Hodrick-Prescott filter for trend-cycle decomposition
# Lambda parameter controls smoothness of the trend component
trend_component, cycle_component = ts.hodrick_prescott_filter(lambda_param=1600)

# Calculate seasonal and noise components
seasonal_noise = prices - trend_component

print("Time series decomposition complete")
```

## Combining Financial and Business Metrics

### Integrated Dashboard for SaaS Business

```python
import numpy as np
from pypulate import as_parray
from pypulate.kpi import (
    churn_rate, retention_rate, customer_lifetime_value,
    customer_acquisition_cost, monthly_recurring_revenue,
    annual_recurring_revenue, ltv_cac_ratio, payback_period
)

# Sample data for a SaaS business
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

# Customer metrics
customers_start = [100, 110, 125, 135, 150, 160]
customers_end = [110, 125, 135, 150, 160, 175]
new_customers = [20, 25, 20, 30, 25, 30]
avg_revenue_per_customer = [50, 52, 53, 55, 56, 58]

# Financial metrics
marketing_costs = [10000, 12000, 11000, 15000, 14000, 16000]
sales_costs = [8000, 9000, 8500, 10000, 9500, 11000]
gross_margin = 70  # 70% gross margin

# Stock price (if public company)
stock_prices = np.array([25, 26, 28, 27, 30, 32])

# Calculate business KPIs
business_metrics = {
    "Month": months,
    "Churn Rate (%)": [],
    "Retention Rate (%)": [],
    "MRR ($)": [],
    "ARR ($)": [],
    "CAC ($)": [],
    "LTV ($)": [],
    "LTV:CAC Ratio": [],
    "Payback Period (months)": []
}

for i in range(len(months)):
    # Calculate metrics
    churn = churn_rate(customers_start[i], customers_end[i], new_customers[i])
    retention = retention_rate(customers_start[i], customers_end[i], new_customers[i])
    mrr = monthly_recurring_revenue(customers_end[i], avg_revenue_per_customer[i])
    arr = annual_recurring_revenue(customers_end[i], avg_revenue_per_customer[i])
    cac = customer_acquisition_cost(marketing_costs[i], sales_costs[i], new_customers[i])
    ltv = customer_lifetime_value(avg_revenue_per_customer[i], gross_margin, churn)
    ratio = ltv_cac_ratio(ltv, cac)
    payback = payback_period(cac, avg_revenue_per_customer[i], gross_margin)
    
    # Store metrics
    business_metrics["Churn Rate (%)"].append(churn)
    business_metrics["Retention Rate (%)"].append(retention)
    business_metrics["MRR ($)"].append(mrr)
    business_metrics["ARR ($)"].append(arr)
    business_metrics["CAC ($)"].append(cac)
    business_metrics["LTV ($)"].append(ltv)
    business_metrics["LTV:CAC Ratio"].append(ratio)
    business_metrics["Payback Period (months)"].append(payback)

# Convert stock prices to Parray for technical analysis
stock_ts = as_parray(stock_prices)

# Calculate technical indicators
stock_sma = stock_ts.sma(3)
stock_ema = stock_ts.ema(3)
stock_rsi = stock_ts.rsi(14)

# Print integrated dashboard
print("=== SaaS Business Dashboard ===")
print("\nBusiness Metrics:")
for i in range(len(months)):
    print(f"\n{months[i]}:")
    print(f"  Churn Rate: {business_metrics['Churn Rate (%)'][i]:.2f}%")
    print(f"  Retention Rate: {business_metrics['Retention Rate (%)'][i]:.2f}%")
    print(f"  MRR: ${business_metrics['MRR ($)'][i]:.2f}")
    print(f"  ARR: ${business_metrics['ARR ($)'][i]:.2f}")
    print(f"  CAC: ${business_metrics['CAC ($)'][i]:.2f}")
    print(f"  LTV: ${business_metrics['LTV ($)'][i]:.2f}")
    print(f"  LTV:CAC Ratio: {business_metrics['LTV:CAC Ratio'][i]:.2f}")
    print(f"  Payback Period: {business_metrics['Payback Period (months)'][i]:.2f} months")

print("\nStock Technical Analysis:")
for i in range(len(months)):
    print(f"\n{months[i]}:")
    print(f"  Stock Price: ${stock_prices[i]:.2f}")
    if i >= 2:  # Need at least 3 points for SMA
        print(f"  SMA(3): ${stock_sma[i]:.2f}")
        print(f"  EMA(3): ${stock_ema[i]:.2f}")
    if i >= 13:  # Need at least 14 points for RSI
        print(f"  RSI(14): {stock_rsi[i]:.2f}")
``` 