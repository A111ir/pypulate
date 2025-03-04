# Getting Started with Pypulate

This guide will help you get started with Pypulate for financial time series analysis, business KPI calculations, and portfolio management.

## Installation

```bash
pip install pypulate
```

## Core Components

Pypulate provides powerful classes for financial and business analytics:

### 1. Parray (Pypulate Array)

The `Parray` class extends NumPy arrays with financial analysis capabilities:

```python
from pypulate import Parray

# Create a price array
prices = Parray([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 15, 11, 8, 10, 14, 16])

# Technical Analysis with method chaining
result = (prices
    .sma(3)                     # Simple Moving Average
    .ema(3)                    # Exponential Moving Average
    .rsi(7)                    # Relative Strength Index
)

# Signal Detection
fast_ma = prices.sma(3)
slow_ma = prices.sma(12)
golden_cross = fast_ma.crossover(slow_ma)
death_cross = fast_ma.crossunder(slow_ma)
```

### 2. KPI (Key Performance Indicators)

The `KPI` class manages business metrics and health assessment:

```python
from pypulate import KPI

# Initialize KPI tracker
kpi = KPI()

# Customer Metrics
churn = kpi.churn_rate(
    customers_start=1000,
    customers_end=950,
    new_customers=50
)

# Financial Metrics
clv = kpi.customer_lifetime_value(
    avg_revenue_per_customer=100,
    gross_margin=70,
    churn_rate_value=5
)

# Health Assessment
health = kpi.health
print(f"Business Health Score: {health['overall_score']}")
print(f"Status: {health['status']}")
```

### 3. Portfolio

The `Portfolio` class handles portfolio analysis and risk management:

```python
from pypulate import Portfolio

# Initialize portfolio analyzer
portfolio = Portfolio()

# Calculate Returns
returns = portfolio.simple_return([50, 100, 120], [60, 70, 120])
twrr = portfolio.time_weighted_return(
    [0.02, 0.01, 0.1, 0.003]
)

# Risk Analysis
sharpe = portfolio.sharpe_ratio(returns, risk_free_rate=0.02)
var = portfolio.value_at_risk(returns, confidence_level=0.95)

# Portfolio Health
health = portfolio.health
print(f"Portfolio Health Score: {health['overall_score']}")
print(f"Risk Status: {health['components']['risk']['status']}")
```

### 4. ServicePricing

The `ServicePricing` class provides a unified interface for various pricing models:

```python
from pypulate import ServicePricing

# Initialize pricing calculator
pricing = ServicePricing()

# Tiered Pricing
price = pricing.calculate_tiered_price(
    usage_units=1500,
    tiers={
        "0-1000": 0.10,    # First tier: $0.10 per unit
        "1001-2000": 0.08, # Second tier: $0.08 per unit
        "2001+": 0.05      # Final tier: $0.05 per unit
    }
)
print(f"Tiered Price: ${price:.2f}")  # $140.00 (1000 * 0.10 + 500 * 0.08)

# Subscription with Features
sub_price = pricing.calculate_subscription_price(
    base_price=99.99,
    features=['premium', 'api_access'],
    feature_prices={'premium': 49.99, 'api_access': 29.99},
    duration_months=12,
    discount_rate=0.10
)

# Track Pricing History
pricing.save_current_pricing()
history = pricing.get_pricing_history()
```

## Common Patterns

### 1. Method Chaining

Parray support method chaining for cleaner code:

```python
# Parray chaining
signals = (Parray(prices)
    .sma(10)
    .crossover(Parray(prices).sma(20))
)
```

### 2. Health Assessments

Portfolio and KPI classes provide health assessments with consistent scoring:

```python
# Business Health
kpi_health = kpi.health  # Business metrics health

# Portfolio Health
portfolio_health = portfolio.health  # Portfolio performance health

# Health Status Categories
# - Excellent: ≥ 90
# - Good: ≥ 75
# - Fair: ≥ 60
# - Poor: ≥ 45
# - Critical: < 45
```

### 3. State Management

All classes maintain state for tracking and analysis:

```python
# KPI state
stored_churn = kpi._state['churn_rate']
stored_retention = kpi._state['retention_rate']

# Portfolio state
stored_returns = portfolio._state['returns']
stored_risk = portfolio._state['volatility']

# ServicePricing state
stored_pricing = pricing._state['current_pricing']
pricing_history = pricing._state['pricing_history']
```

## Next Steps

Now that you understand the basic components, explore these topics in detail:

- [Parray Guide](parray.md): Advanced technical analysis and signal detection
- [KPI Guide](kpi.md): Comprehensive business metrics and health scoring
- [Portfolio Guide](portfolio.md): Portfolio analysis and risk management
- [Service Pricing Guide](service-pricing.md): Pricing models and calculations