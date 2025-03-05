# Pypulate

![Pypulate Logo](docs/assets/logo.png)

[![PyPI](https://img.shields.io/badge/pypi-v0.2.1-blue)](https://pypi.org/project/pypulate/)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-passing-brightgreen)
![Documentation](https://img.shields.io/badge/docs-latest-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI Downloads](https://static.pepy.tech/badge/pypulate)](https://pepy.tech/projects/pypulate)
> **High-performance financial and business analytics framework for Python**

Pypulate is a comprehensive Python framework designed for financial analysis, business metrics tracking, portfolio management, and service pricing. It provides powerful tools for quantitative analysts, business analysts, and financial professionals to analyze data, track KPIs, manage portfolios, and implement pricing strategies.

## ✨ Features

### Parray (Pypulate Array)
- Technical indicators (30+ implementations)
- Signal detection and pattern recognition
- Time series transformations
- Built-in filtering methods
- Method chaining support

### KPI (Key Performance Indicators)
- Customer metrics (churn, retention, LTV)
- Financial metrics (ROI, CAC, ARR)
- Engagement metrics (NPS, CSAT)
- Health scoring system
- Metric tracking and history

### Portfolio Management
- Return calculations (simple, log, time-weighted)
- Risk metrics (Sharpe, VaR, drawdown)
- Performance attribution
- Portfolio health assessment
- Risk management tools

### Portfolio Allocation
- Mean-Variance Optimization
- Risk Parity Portfolio
- Kelly Criterion (with half-Kelly option)
- Black-Litterman model
- Hierarchical Risk Parity
- Custom constraints support
- Multiple optimization methods

### Service Pricing
- Tiered pricing models
- Subscription pricing with features
- Usage-based pricing
- Dynamic pricing adjustments
- Volume discounts
- Custom pricing rules
- Pricing history tracking

## 🚀 Installation

```bash
pip install pypulate
```

## 🔧 Quick Start

### Technical Analysis
```python
from pypulate import Parray

# Create a price array
prices = Parray([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 15, 11])

# Technical Analysis with method chaining
result = (prices
    .sma(3)                    # Simple Moving Average
    .ema(3)                    # Exponential Moving Average
    .rsi(7)                    # Relative Strength Index
)

# Signal Detection
golden_cross = prices.sma(5).crossover(prices.sma(10))
```

### Business KPIs
```python
from pypulate import KPI

kpi = KPI()

# Calculate Customer Metrics
churn = kpi.churn_rate(
    customers_start=1000,
    customers_end=950,
    new_customers=50
)

# Get Business Health
health = kpi.health
print(f"Business Health Score: {health['overall_score']}")
```

### Portfolio Analysis
```python
from pypulate import Portfolio

portfolio = Portfolio()

# Calculate Returns and Risk
returns = portfolio.simple_return([100, 102, 105], [102, 105, 108])
sharpe = portfolio.sharpe_ratio(returns, risk_free_rate=0.02)
var = portfolio.value_at_risk(returns, confidence_level=0.95)

# Get Portfolio Health
health = portfolio.health
print(f"Portfolio Health: {health['status']}")
```

### Portfolio Allocation
```python
from pypulate import Allocation
import numpy as np

allocation = Allocation()

# Sample returns data (252 days, 5 assets)
returns = np.random.normal(0.0001, 0.02, (252, 5))
risk_free_rate = 0.04

# Mean-Variance Optimization
weights, ret, risk = allocation.mean_variance(
    returns, 
    risk_free_rate=risk_free_rate
)
print(f"Mean-Variance Portfolio:")
print(f"Expected Return: {ret:.2%}")
print(f"Risk: {risk:.2%}")

# Risk Parity Portfolio
weights, ret, risk = allocation.risk_parity(returns)

# Kelly Criterion (with half-Kelly)
weights, ret, risk = allocation.kelly_criterion(
    returns, 
    kelly_fraction=0.5
)

# Black-Litterman with views
views = {0: 0.15, 1: 0.12}  # Views on first two assets
view_confidences = {0: 0.8, 1: 0.7}
market_caps = np.array([1000, 800, 600, 400, 200])
weights, ret, risk = allocation.black_litterman(
    returns, market_caps, views, view_confidences
)
```

### Service Pricing
```python
from pypulate import ServicePricing

pricing = ServicePricing()

# Calculate Tiered Price
price = pricing.calculate_tiered_price(
    usage_units=1500,
    tiers={
        "0-1000": 0.10,
        "1001-2000": 0.08,
        "2001+": 0.05
    }
)

# Calculate Subscription Price
sub_price = pricing.calculate_subscription_price(
    base_price=99.99,
    features=['premium', 'api_access'],
    feature_prices={'premium': 49.99, 'api_access': 29.99},
    duration_months=12,
    discount_rate=0.10
)
```

## 📊 Key Capabilities

### Data Analysis
- Time series analysis and transformations
- Technical indicators and signal detection
- Pattern recognition
- Performance metrics

### Business Analytics
- Customer analytics
- Financial metrics
- Health scoring
- Metric tracking and history

### Risk Management
- Portfolio optimization
- Risk assessment
- Performance attribution
- Health monitoring
- Asset allocation strategies
- Multiple optimization methods

### Pricing Strategies
- Multiple pricing models
- Dynamic adjustments
- Custom rule creation
- History tracking

## 📚 Documentation

Comprehensive documentation is available at [https://a111ir.github.io/pypulate](https://a111ir.github.io/pypulate) or in the docs directory:

- [Getting Started Guide](https://a111ir.github.io/pypulate/user-guide/getting-started/)
- [Parray Guide](https://a111ir.github.io/pypulate/user-guide/parray/)
- [KPI Guide](https://a111ir.github.io/pypulate/user-guide/kpi/)
- [Portfolio Guide](https://a111ir.github.io/pypulate/user-guide/portfolio/)
- [Service Pricing Guide](https://a111ir.github.io/pypulate/user-guide/service-pricing/)
- [Allocation Guide](https://a111ir.github.io/pypulate/user-guide/allocation/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  Made with ❤️ for financial and business analytics
</p>
