# Portfolio Allocation Examples

This document demonstrates various portfolio optimization methods available in the `pypulate` library using the `Allocation` class.

## Overview

The `Allocation` class provides several portfolio optimization methods:

1. Mean-Variance Optimization
2. Minimum Variance Portfolio
3. Maximum Sharpe Ratio Portfolio
4. Risk Parity Portfolio
5. Maximum Diversification Portfolio
6. Equal Weight Portfolio
7. Market Cap Weight Portfolio
8. Kelly Criterion Portfolio
9. Black-Litterman Portfolio
10. Hierarchical Risk Parity Portfolio

## Basic Usage

First, let's import the necessary components:

```python
from pypulate import Allocation, Parray
```

## Basic Portfolio Optimization

Let's start with a simple example using 3 assets:

```python
# Sample returns data (3 assets, 252 days of daily returns)
returns = Parray([
    # Asset 1 (e.g., AAPL)
    [0.02, -0.01, 0.015, 0.03, -0.005, 0.01, 0.02, -0.015, 0.025, 0.01] + 
    [0.015, -0.02, 0.01, 0.02, -0.01, 0.015, 0.025, -0.01, 0.02, 0.015] * 24 + 
    [0.015, -0.02, 0.01, 0.02, -0.01, 0.015, 0.025, -0.01, 0.02, 0.015],
    
    # Asset 2 (e.g., MSFT)
    [0.015, 0.02, -0.01, 0.025, 0.01, -0.015, 0.02, 0.01, -0.02, 0.015] + 
    [0.02, -0.015, 0.015, 0.025, -0.01, 0.02, 0.015, -0.015, 0.02, 0.01] * 24 + 
    [0.02, -0.015, 0.015, 0.025, -0.01, 0.02, 0.015, -0.015, 0.02, 0.01],
    
    # Asset 3 (e.g., GOOGL)
    [0.025, -0.02, 0.02, 0.015, -0.015, 0.02, 0.025, -0.02, 0.03, 0.02] + 
    [0.025, -0.02, 0.02, 0.015, -0.015, 0.02, 0.025, -0.02, 0.03, 0.02] * 24 + 
    [0.025, -0.02, 0.02, 0.015, -0.015, 0.02, 0.025, -0.02, 0.03, 0.02]
]).T

# Initialize the Allocation class
allocation = Allocation()

# Set risk-free rate (e.g., current 10-year Treasury yield)
risk_free_rate = 0.04  # 4% annual rate

# Perform Mean-Variance Optimization
weights, ret, risk = allocation.mean_variance(
    returns=returns,
    target_return=None,  # Maximize Sharpe ratio
    risk_free_rate=risk_free_rate
)

print(f"Optimal Portfolio Weights: {weights}")
print(f"Expected Return: {ret:.4f}")
print(f"Portfolio Risk: {risk:.4f}")
```

## Risk Parity Portfolio

Risk Parity aims to equalize the risk contribution of each asset:

```python
# Calculate Risk Parity weights
weights, ret, risk = allocation.risk_parity(returns=returns)

print(f"Risk Parity Weights: {weights}")
print(f"Expected Return: {ret:.4f}")
print(f"Portfolio Risk: {risk:.4f}")
```

## Kelly Criterion with Conservative Sizing

The Kelly Criterion can be aggressive, so we often use a fraction of the optimal weights:

```python
# Calculate Kelly Criterion weights
weights, ret, risk = allocation.kelly_criterion(
    returns=returns,
    risk_free_rate=risk_free_rate
)

# Use half-Kelly for more conservative position sizing
half_kelly_weights = weights * 0.5

print(f"Full Kelly Weights: {weights}")
print(f"Half-Kelly Weights: {half_kelly_weights}")
```

## Black-Litterman Portfolio with Views

Black-Litterman allows incorporating market views into the optimization:

```python
# Market capitalizations
market_caps = Parray([2.5e12, 2.8e12, 1.8e12])  # in USD

# Define views (e.g., AAPL to outperform by 2%, GOOGL to underperform by 1%)
views = {0: 0.02, 2: -0.01}  # Asset indices and expected excess returns
view_confidences = {0: 0.8, 2: 0.7}  # Confidence in views (0-1)

# Calculate Black-Litterman weights
weights, ret, risk = allocation.black_litterman(
    returns=returns,
    market_caps=market_caps,
    views=views,
    view_confidences=view_confidences,
    tau=0.05,  # Uncertainty in the prior distribution
    risk_free_rate=risk_free_rate
)

print(f"Black-Litterman Weights: {weights}")
```

## Hierarchical Risk Parity

HRP uses hierarchical clustering to build a more robust portfolio:

```python
# Calculate HRP weights
weights, ret, risk = allocation.hierarchical_risk_parity(
    returns=returns,
    linkage_method='ward',  # Using Ward linkage for clustering
    distance_metric='correlation'  # Using correlation-based distance
)

print(f"HRP Weights: {weights}")
```

## Comparing Different Methods

Here's how to compare the performance of different optimization methods:

```python
# Define methods to compare
methods = [
    ("Mean-Variance", allocation.mean_variance(returns, risk_free_rate=risk_free_rate)),
    ("Minimum Variance", allocation.minimum_variance(returns)),
    ("Maximum Sharpe", allocation.maximum_sharpe(returns, risk_free_rate=risk_free_rate)),
    ("Risk Parity", allocation.risk_parity(returns)),
    ("Kelly Criterion", allocation.kelly_criterion(returns, risk_free_rate=risk_free_rate))
]

# Compare results
print("\nMethod Comparison Summary:")
print("-" * 50)
print(f"{'Method':<25} {'Return':>10} {'Risk':>10} {'Sharpe':>10}")
print("-" * 50)
for method_name, (weights, ret, risk) in methods:
    sharpe = (ret - risk_free_rate) / risk
    print(f"{method_name:<25} {ret*100:>9.2f}% {risk*100:>9.2f}% {sharpe:>9.2f}")
```

## Best Practices

1. **Data Quality**:
   - Use clean, adjusted price data
   - Handle missing values appropriately
   - Consider transaction costs and liquidity

2. **Risk Management**:
   - Consider using half-Kelly or quarter-Kelly for more conservative position sizing
   - Implement position limits and constraints
   - Monitor portfolio turnover

3. **Method Selection**:
   - Mean-Variance: Good for traditional portfolio optimization
   - Risk Parity: Better for risk management
   - Kelly Criterion: Best for long-term growth
   - Black-Litterman: Ideal when you have strong market views
   - HRP: More robust to estimation errors

4. **Regular Rebalancing**:
   - Set appropriate rebalancing thresholds
   - Consider transaction costs
   - Monitor tracking error

## Common Pitfalls

1. **Overfitting**:
   - Use sufficient historical data
   - Consider using rolling windows
   - Implement out-of-sample testing

2. **Estimation Error**:
   - Use robust estimation methods
   - Consider shrinkage estimators
   - Implement proper regularization

3. **Transaction Costs**:
   - Account for bid-ask spreads
   - Consider market impact
   - Implement turnover constraints
