# Pypulate

![Pypulate Logo](docs/assets/logo.png)

[![PyPI](https://img.shields.io/badge/pypi-v0.3.0-blue)](https://pypi.org/project/pypulate/)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-100%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-passing-brightgreen)
![Documentation](https://img.shields.io/badge/docs-latest-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI Downloads](https://static.pepy.tech/badge/pypulate)](https://pepy.tech/projects/pypulate)
> **High-performance financial and business analytics framework for Python**

Pypulate is a comprehensive Python framework designed for financial analysis, business metrics tracking, portfolio management, and service pricing. It provides powerful tools for quantitative analysts, business analysts, and financial professionals to analyze data, track KPIs, manage portfolios, and implement pricing strategies.

## ✨ Features

### Parray (Pypulate Array)
- **Preprocessing capabilities**:
  - Normalization & scaling (L1/L2, min-max, robust scaling)
  - Outlier detection & handling (z-score, IQR, winsorization)
  - Missing value imputation (mean, median, forward/backward fill)
  - Statistical transformations (log, power, tanh transforms)
  - Signal filtering (Kalman, Butterworth, Savitzky-Golay)
  - Performance optimization (GPU acceleration, parallel processing)
- Technical indicators (30+ implementations)
- Signal detection and pattern recognition
- Time series transformations
- Built-in filtering methods
- Method chaining support

### Data Processing
- Normalization and scaling methods
- Outlier detection and handling
- Missing value imputation
- Time series transformations
- Statistical analysis tools
- Correlation and covariance analysis
- Stationarity and causality tests

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
- Freemium pricing
- Loyalty pricing
- Volume discounts
- Custom pricing rules
- Pricing history tracking

### Credit Scoring
- Bankruptcy prediction models (Altman Z-Score)
- Default probability estimation (Merton model)
- Credit scorecard development
- Financial ratio analysis
- Expected credit loss calculation (IFRS 9/CECL)
- Risk-based loan pricing
- Credit model validation
- Loss given default estimation
- Exposure at default calculation
- Credit rating transition matrices

### Asset Pricing
- Capital Asset Pricing Model (CAPM)
- Arbitrage Pricing Theory (APT)
- Fama-French factor models (3-factor and 5-factor)
- Black-Scholes option pricing
- Implied volatility calculation
- Binomial tree option pricing
- Bond pricing and yield calculations
- Duration and convexity analysis
- Yield curve construction
- Term structure models

## 🚀 Installation

```bash
pip install pypulate
```

## 🔧 Quick Start

### Technical Analysis with Preprocessing
```python
from pypulate import Parray
import numpy as np

# Create a price array with some outliers and missing values
prices = Parray([10, 11, 12, np.nan, 10, 50, 10, 11, 12, 13, 15, np.nan, 8, 10, 14, 16])

# Data Preprocessing Pipeline
clean_prices = (prices
    .fill_missing(method='forward')                  # Fill missing values
    .remove_outliers(method='zscore', threshold=3.0) # Remove statistical outliers
    .standardize()                                   # Z-score standardization
)

# Performance Optimization
clean_prices.enable_parallel(num_workers=4)  # Enable parallel processing
if Parray.is_gpu_available():
    clean_prices.enable_gpu()                # Use GPU if available
clean_prices.optimize_memory()               # Optimize memory usage

# Technical Analysis with method chaining on cleaned data
result = (clean_prices
    .sma(3)                    # Simple Moving Average
    .ema(3)                    # Exponential Moving Average
    .rsi(7)                    # Relative Strength Index
)

# Signal Detection
golden_cross = clean_prices.sma(5).crossover(clean_prices.sma(10))
```

### More Preprocessing Examples
```python
from pypulate import Parray
import numpy as np

# Create sample data
data = Parray(np.random.normal(0, 1, 1000))
data_with_outliers = Parray([10, 11, 12, 11, 10, 9, 50, 11, 12, 13, -40, 11, 8, 10, 14, 16])

# Normalization and Scaling
normalized = data.normalize(method='l2')              # L2 normalization
standardized = data.standardize()                     # Z-score standardization
scaled = data.min_max_scale(feature_range=(0, 1))     # Min-max scaling
robust_scaled = data.robust_scale(method='iqr')       # Robust scaling using IQR

# Outlier Handling
clean_data = data_with_outliers.remove_outliers(method='zscore', threshold=3.0)  # Remove outliers
clipped_data = data_with_outliers.clip_outliers(lower_percentile=1.0, upper_percentile=99.0)
winsorized = data_with_outliers.winsorize(limits=0.1)  # Winsorize at 10% on both ends

# Data Transformations
power_data = data.power_transform(method='yeo-johnson')  # Power transformation
discretized = data.discretize(n_bins=5, strategy='quantile')  # Discretization

# Signal Filtering
kalman_filtered = data.kalman_filter(process_variance=1e-5, measurement_variance=1e-3)
butter_filtered = data.butterworth_filter(cutoff=0.1, order=2, filter_type='lowpass')
sg_filtered = data.savitzky_golay_filter(window_length=11, polyorder=3)

# Method Chaining for Complex Pipelines
result = (data_with_outliers
    .fill_missing(method='median')          # Fill any missing values with median
    .clip_outliers(lower_percentile=5.0, upper_percentile=95.0)  # Clip extreme values
    .standardize()                          # Standardize to mean=0, std=1
    .butterworth_filter(cutoff=0.1, order=3, filter_type='lowpass')  # Apply smoothing
)
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

### Credit Scoring
```python
from pypulate.dtypes import CreditScoring

credit = CreditScoring()

# Corporate Credit Risk Assessment
z_score = credit.altman_z_score(
    working_capital=1200000,
    retained_earnings=1500000,
    ebit=800000,
    market_value_equity=5000000,
    sales=4500000,
    total_assets=6000000,
    total_liabilities=2500000
)
print(f"Altman Z-Score: {z_score['z_score']:.2f}")
print(f"Risk Assessment: {z_score['risk_assessment']}")

# Retail Credit Scoring
scorecard_result = credit.create_scorecard(
    features={
        "age": 35,
        "income": 75000,
        "years_employed": 5,
        "debt_to_income": 0.3
    },
    weights={
        "age": 2.5,
        "income": 3.2,
        "years_employed": 4.0,
        "debt_to_income": -5.5
    }
)
print(f"Credit Score: {scorecard_result['total_score']:.0f}")
print(f"Risk Category: {scorecard_result['risk_category']}")

# Expected Credit Loss
ecl = credit.expected_credit_loss(
    pd=0.05,  # Probability of default
    lgd=0.4,  # Loss given default
    ead=100000  # Exposure at default
)
print(f"Expected Credit Loss: ${ecl['ecl']:.2f}")
```

### Asset Pricing
```python
from pypulate.asset import capm, black_scholes, price_bond

# CAPM Example
result = capm(
    risk_free_rate=0.03,
    beta=1.2,
    market_return=0.08
)
print(f"Expected Return: {result['expected_return']:.2%}")  # 9.00%

# Black-Scholes Option Pricing
option = black_scholes(
    option_type='call',
    underlying_price=100,
    strike_price=100,
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    volatility=0.2
)
print(f"Call Option Price: ${option['price']:.2f}")  # $10.45

# Bond Pricing
bond = price_bond(
    face_value=1000,
    coupon_rate=0.05,
    years_to_maturity=10,
    yield_to_maturity=0.06,
    frequency=2
)
print(f"Bond Price: ${bond['price']:.2f}")  # $925.68
```

## 📊 Key Capabilities

### Data Preprocessing
- **Comprehensive preprocessing** methods for financial time series
- **Outlier handling** with multiple detection methods (statistical, distance-based)
- **Missing value imputation** strategies optimized for time series
- **Normalization and scaling** to improve model performance
- **Statistical transformations** to handle non-normal data distributions
- **Signal filtering** to remove noise from financial data
- **Performance optimization** with GPU acceleration and parallel processing
- **Memory optimization** for handling large datasets

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

### Credit Risk Assessment
- Bankruptcy prediction
- Default probability modeling
- Credit scoring and scorecards
- Financial ratio analysis
- Expected credit loss calculation
- Risk-based loan pricing
- Credit model validation

### Asset Pricing and Derivatives
- Equity pricing models
- Option pricing and Greeks
- Fixed income analysis
- Yield curve modeling
- Risk-neutral valuation

## 📚 Documentation

Comprehensive documentation is available at [https://a111ir.github.io/pypulate](https://a111ir.github.io/pypulate) or in the docs directory:

- [Getting Started Guide](https://a111ir.github.io/pypulate/user-guide/getting-started/)
- [Parray Guide](https://a111ir.github.io/pypulate/user-guide/parray/)
- [Preprocessing Guide](https://a111ir.github.io/pypulate/user-guide/preprocessing/)
- [KPI Guide](https://a111ir.github.io/pypulate/user-guide/kpi/)
- [Portfolio Guide](https://a111ir.github.io/pypulate/user-guide/portfolio/)
- [Service Pricing Guide](https://a111ir.github.io/pypulate/user-guide/service-pricing/)
- [Allocation Guide](https://a111ir.github.io/pypulate/user-guide/allocation/)
- [Credit Scoring Guide](https://a111ir.github.io/pypulate/user-guide/credit-scoring/)
- [Asset Pricing Guide](https://a111ir.github.io/pypulate/user-guide/asset-pricing/)

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
