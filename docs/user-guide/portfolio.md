# Portfolio Guide

The `Portfolio` class provides a comprehensive suite of methods for portfolio analysis, risk management, and performance attribution.

## Basic Usage

```python
from pypulate import Portfolio

# Initialize portfolio analyzer
portfolio = Portfolio()

# Calculate basic returns
simple_ret = portfolio.simple_return(10, 12)
log_ret = portfolio.log_return(10, 12)


```

## Return Metrics

### Simple Returns
```python
# Calculate simple returns
returns = portfolio.simple_return([10, 20, 25], [12, 21, 20])

# Calculate holding period return
hpr = portfolio.holding_period_return(
    [10, 20, 25]
)

# Calculate annualized return
annual_ret = portfolio.annualized_return(
    [10, 20, 25],
    years=2
)
```

### Time-Weighted Returns
```python
# Calculate time-weighted return
twrr = portfolio.time_weighted_return(
    [0.01, 0.03, 0.02, 0.02, 0.001],
)
```

### Money-Weighted Returns
```python
# Calculate money-weighted return (IRR)
mwrr = portfolio.money_weighted_return([-1000, -500, 1700], [0, 0.5, 1], 0)
```

## Risk Metrics

### Volatility Measures
```python
# Calculate standard deviation
std_dev = portfolio.standard_deviation([0.01, 0.03, 0.02, 0.02, 0.001])
```

### Value at Risk
```python
# Calculate parametric VaR
var = portfolio.value_at_risk(
    [0.01, 0.03, 0.02, 0.02, 0.001],
    confidence_level=0.95,
    method = 'monte_carlo'
)

# Calculate conditional VaR (Expected Shortfall)
cvar = portfolio.conditional_value_at_risk(
    returns,
    confidence_level=0.95
)
```

### Drawdown Analysis
```python
# Calculate maximum drawdown
max_dd = portfolio.max_drawdown(prices)

# Get drawdown details
dd_amount, dd_percent, dd_length = portfolio.drawdown_details(prices)
```

## Risk-Adjusted Performance

### Sharpe Ratio
```python
# Calculate Sharpe ratio
sharpe = portfolio.sharpe_ratio(
    returns,
    risk_free_rate=0.02,
    periods_per_year=252
)
```

### Information Ratio
```python
# Calculate Information ratio
info_ratio = portfolio.information_ratio(
    returns,
    benchmark_returns,
    periods_per_year=252
)
```

### CAPM Metrics
```python
# Calculate beta
beta = portfolio.beta(returns, market_returns)

# Calculate alpha
alpha = portfolio.alpha(
    returns,
    market_returns,
    risk_free_rate=0.02
)
```

## Health Assessment

The `health` property provides a comprehensive assessment of portfolio health:

```python
health = portfolio.health

# Health assessment structure
{
    'overall_score': 82.5,
    'status': 'Good',
    'components': {
        'returns': {
            'score': 85.0,
            'status': 'Good'
        },
        'risk': {
            'score': 78.0,
            'status': 'Good'
        },
        'risk_adjusted': {
            'score': 88.0,
            'status': 'Good'
        }
    }
}
```

### Health Score Components

The portfolio health score is calculated based on three main components:

- **Returns (30%)**
  - Absolute Returns
  - Relative Returns
  - Consistency of Returns

- **Risk Metrics (40%)**
  - Volatility
  - Value at Risk
  - Maximum Drawdown
  - Recovery Time

- **Risk-Adjusted Performance (30%)**
  - Sharpe Ratio
  - Information Ratio
  - Sortino Ratio
  - Treynor Ratio

Each component is scored from 0-100 and assigned a status:
- Excellent: ≥ 90
- Good: ≥ 75
- Fair: ≥ 60
- Poor: ≥ 45
- Critical: < 45

## State Management

The Portfolio class maintains state for calculated metrics in the `_state` dictionary:

```python
# Access stored metrics
stored_returns = portfolio._state['returns']
stored_volatility = portfolio._state['volatility']
stored_sharpe = portfolio._state['sharpe_ratio']
```

## Best Practices

1. **Data Quality**: Ensure price and return data is clean and properly formatted
2. **Time Periods**: Use consistent time periods for comparative analysis
3. **Risk Management**: Regularly monitor risk metrics
4. **Benchmark Selection**: Choose appropriate benchmarks for relative analysis
5. **Health Monitoring**: Regularly assess portfolio health 