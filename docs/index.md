# Pypulate Documentation

![Pypulate Logo](assets/logo.png)

!!! tip "Pypulate framework"
    **High performance financial and business analytics framework**

    ![GitHub Workflow Status](https://img.shields.io/badge/tests-passing-brightgreen)
    ![Coverage](https://img.shields.io/badge/coverage-passing-brightgreen)
    ![PyPI](https://img.shields.io/badge/pypi-v0.1.0-brightgreen)
    ![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

Welcome to Pypulate, a high-performance Python framework for financial analysis and business metrics. Pypulate offers powerful classes designed to handle different aspects of financial and business analytics (more to come):

## Core Components

### Parray (Pypulate Array)

A specialized array class for financial time series analysis with built-in technical analysis capabilities and chain methodss:

```python
from pypulate import Parray

# Create a price array
prices = Parray([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 15, 11, 8, 10, 14, 16])

# Technical Analysis
sma = prices.sma(5)                   
rsi = prices.rsi(14)                  
bb_upper, bb_mid, bb_lower = prices.bollinger_bands(20, 2)

# Signal Detection
golden_cross = prices.sma(5).crossover(prices.sma(10))
death_cross = prices.sma(5).crossunder(prices.sma(10))

# Volatility Analysis
volatility = prices.historical_volatility(7)
```

### KPI (Key Performance Indicators)

A comprehensive class for calculating and tracking business metrics:

```python
from pypulate import KPI

kpi = KPI()

# Customer Metrics
churn = kpi.churn_rate(customers_start=1000, customers_end=950, new_customers=50)
retention = kpi.retention_rate(customers_start=1000, customers_end=950, new_customers=50)

# Financial Metrics
clv = kpi.customer_lifetime_value(
    avg_revenue_per_customer=100,
    gross_margin=70,
    churn_rate_value=5
)

# Health Assessment
health = kpi.health  # Returns overall business health score and component analysis
```

### Portfolio

A class for portfolio analysis and risk management:

```python
from pypulate import Portfolio
import numpy as np

portfolio = Portfolio()

# Sample data
start = [100, 102, 105, 103, 107, 108, 107, 110, 112, 111]
end = [110, 95, 111, 103, 130, 89, 99, 104, 102, 100]

cash_flows = [0, -1000, 0, 500, 0, -2000, 0, 1000, 0, 0]
risk_free_rate = 0.02

# Return Metrics
returns = portfolio.simple_return(end, start)
log_ret = portfolio.log_return(end, start)

# Risk Metrics
sharpe = portfolio.sharpe_ratio(returns, risk_free_rate)
var = portfolio.value_at_risk(returns, confidence_level=0.95)
dd = portfolio.drawdown(returns)

# Portfolio Health
health = portfolio.health  # Returns portfolio health analysis
```

### ServicePricing

A unified interface for calculating and managing various service pricing models:

```python
from pypulate import ServicePricing

pricing = ServicePricing()

# Tiered Pricing
price = pricing.calculate_tiered_price(
    usage_units=1500,
    tiers={
        "0-1000": 0.10,    # $0.10 per unit for first 1000 units
        "1001-2000": 0.08, # $0.08 per unit for next 500 units
        "2001+": 0.05      # $0.05 per unit for remaining units
    }
)

# Subscription with Features
sub_price = pricing.calculate_subscription_price(
    base_price=99.99,
    features=['premium', 'api_access'],
    feature_prices={'premium': 49.99, 'api_access': 29.99},
    duration_months=12,
    discount_rate=0.10
)

# Dynamic Pricing
dynamic_price = pricing.apply_dynamic_pricing(
    base_price=100.0,
    demand_factor=1.2,
    competition_factor=0.9,
    seasonality_factor=1.1
)

# Track Pricing History
pricing.save_current_pricing()
history = pricing.get_pricing_history()
```

## Installation

```bash
pip install pypulate
```

## Key Features

- **Parray**: 
  - Technical indicators (30+ implementations)
  - Signal detection and pattern recognition
  - Time series transformations
  - Built-in filtering methods

- **KPI**:
  - Customer metrics (churn, retention, LTV)
  - Financial metrics (ROI, CAC, ARR)
  - Engagement metrics (NPS, CSAT)
  - Health scoring system

- **Portfolio**:
  - Return calculations
  - Risk metrics
  - Performance attribution
  - Health assessment

- **ServicePricing**:
  - Tiered pricing models
  - Subscription pricing with features
  - Usage-based pricing
  - Dynamic pricing adjustments
  - Volume discounts
  - Custom pricing rules
  - Pricing history tracking

## User Guide

- [Getting Started](user-guide/getting-started.md)
- [Parray Guide](user-guide/parray.md)
- [KPI Guide](user-guide/kpi.md)
- [Portfolio Guide](user-guide/portfolio.md)
- [Service Pricing Guide](user-guide/service-pricing.md)


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
