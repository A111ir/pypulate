# Getting Started with Pypulate

This guide will help you get started with Pypulate for financial time series analysis, business KPI calculations, and portfolio management.

## Installation

```bash
pip install pypulate
```

## Core Components

Pypulate provides powerful classes for financial and business analytics:

### 1. Parray (Pypulate Array)

The `Parray` class extends NumPy arrays with financial analysis capabilities, preprocessing functions, and performance optimizations:

```python
from pypulate import Parray
import numpy as np

# Create a price array from various sources
prices = Parray([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 15, 11, 8, 10, 14, 16])
prices_from_numpy = Parray(np.random.normal(100, 5, 1000))

# Technical Analysis with method chaining
result = (prices
    .sma(3)                    # Simple Moving Average
    .ema(3)                    # Exponential Moving Average
    .rsi(7)                    # Relative Strength Index
)

# Data Preprocessing
clean_data = (prices
    .remove_outliers(method='zscore', threshold=3.0)  # Remove statistical outliers
    .fill_missing(method='forward')                  # Fill missing values with forward fill
    .interpolate_missing(method='cubic')             # Interpolate remaining missing values
)

# Standardization and Normalization
normalized_data = prices.normalize(method='l2')              # L2 normalization
standardized_data = prices.standardize()                     # Z-score standardization
scaled_data = prices.min_max_scale(feature_range=(0, 1))     # Min-max scaling
robust_data = prices.robust_scale(method='iqr')              # Robust scaling using IQR

# Transformations
log_data = prices.log_transform(offset=1.0)                  # Log transformation
power_transformed = prices.power_transform(method='yeo-johnson')  # Power transformation
discretized = prices.discretize(n_bins=5, strategy='quantile')  # Discretization

# Time Series Operations
lagged_features = prices.lag_features(lags=[1, 2, 3])        # Create lag features
resampled = prices.resample(factor=2, method='mean')         # Downsample data

# Signal Detection
fast_ma = prices.sma(3)
slow_ma = prices.sma(12)
golden_cross = fast_ma.crossover(slow_ma)
death_cross = fast_ma.crossunder(slow_ma)

# Statistical Tests
adf_stat, adf_pvalue = prices.augmented_dickey_fuller_test()  # Test for stationarity
stats = prices.descriptive_stats()                           # Get descriptive statistics
```

#### Key Features of Parray

1. **Performance Optimization**
    - GPU Acceleration: Use `enable_gpu()` for faster computations on compatible hardware
    - Parallel Processing: Use `enable_parallel()` to utilize multiple CPU cores
    - Memory Optimization: Use `optimize_memory()` for efficient memory usage with large datasets

2. **Data Preprocessing**
    - Outlier Handling: `remove_outliers()`, `winsorize()`, `clip_outliers()`
    - Missing Value Handling: `fill_missing()`, `interpolate_missing()`
    - Normalization: `normalize()`, `standardize()`, `min_max_scale()`, `robust_scale()`
    - Transformations: `log_transform()`, `power_transform()`, `dynamic_tanh()`

3. **Technical Analysis**
    - Moving Averages: `sma()`, `ema()`, `wma()`, `hma()`, `zlma()`, etc.
    - Momentum Indicators: `rsi()`, `macd()`, `stochastic_oscillator()`, etc.
    - Volatility Indicators: `bollinger_bands()`, `atr()`, `keltner_channels()`, etc.
    - Signal Detection: `crossover()`, `crossunder()`

4. **Method Chaining**
    - Chain methods together for complex operations in a single, readable line
    - Avoid creating intermediate variables
    - Efficiently combine preprocessing, analysis, and signal generation

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

### 4. Allocation

The `Allocation` class provides advanced portfolio optimization and asset allocation methods:

```python
from pypulate import Allocation
import numpy as np

# Initialize allocation optimizer
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
print(f"Weights: {weights}")

# Risk Parity Portfolio
weights, ret, risk = allocation.risk_parity(returns)
print(f"\nRisk Parity Portfolio:")
print(f"Expected Return: {ret:.2%}")
print(f"Risk: {risk:.2%}")
print(f"Weights: {weights}")

# Kelly Criterion (with half-Kelly)
weights, ret, risk = allocation.kelly_criterion(
    returns, 
    kelly_fraction=0.5
)
print(f"\nHalf-Kelly Portfolio:")
print(f"Expected Return: {ret:.2%}")
print(f"Risk: {risk:.2%}")
print(f"Weights: {weights}")

# Black-Litterman with views
views = {0: 0.15, 1: 0.12}  # Views on first two assets
view_confidences = {0: 0.8, 1: 0.7}
market_caps = np.array([1000, 800, 600, 400, 200])
weights, ret, risk = allocation.black_litterman(
    returns, 
    market_caps, 
    views, 
    view_confidences
)
print(f"\nBlack-Litterman Portfolio:")
print(f"Expected Return: {ret:.2%}")
print(f"Risk: {risk:.2%}")
print(f"Weights: {weights}")

# Hierarchical Risk Parity
weights, ret, risk = allocation.hierarchical_risk_parity(returns)
print(f"\nHierarchical Risk Parity Portfolio:")
print(f"Expected Return: {ret:.2%}")
print(f"Risk: {risk:.2%}")
print(f"Weights: {weights}")
```

### 5. ServicePricing

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

### 6. CreditScoring

The `CreditScoring` class provides comprehensive credit risk assessment and scoring tools:

```python
from pypulate.dtypes import CreditScoring

# Initialize credit scoring system
credit = CreditScoring()

# Corporate Credit Risk Assessment
z_score_result = credit.altman_z_score(
    working_capital=1200000,
    retained_earnings=1500000,
    ebit=800000,
    market_value_equity=5000000,
    sales=4500000,
    total_assets=6000000,
    total_liabilities=2500000
)
print(f"Altman Z-Score: {z_score_result['z_score']:.2f}")
print(f"Risk Assessment: {z_score_result['risk_assessment']}")

# Default Probability Estimation
merton_result = credit.merton_model(
    asset_value=10000000,
    debt_face_value=5000000,
    asset_volatility=0.25,
    risk_free_rate=0.03,
    time_to_maturity=1.0
)
print(f"Probability of Default: {merton_result['probability_of_default']:.2%}")

# Credit Scorecard for Retail Lending
features = {
    "age": 35,
    "income": 75000,
    "years_employed": 5,
    "debt_to_income": 0.3,
    "previous_defaults": 0
}
weights = {
    "age": 2.5,
    "income": 3.2,
    "years_employed": 4.0,
    "debt_to_income": -5.5,
    "previous_defaults": -25.0
}
scorecard_result = credit.create_scorecard(
    features=features,
    weights=weights,
    scaling_factor=20,
    base_score=600
)
print(f"Credit Score: {scorecard_result['total_score']:.0f}")
print(f"Risk Category: {scorecard_result['risk_category']}")

# Expected Credit Loss Calculation
ecl_result = credit.expected_credit_loss(
    pd=0.05,  # Probability of default
    lgd=0.4,  # Loss given default
    ead=100000,  # Exposure at default
    time_horizon=1.0
)
print(f"Expected Credit Loss: ${ecl_result['ecl']:.2f}")

# Track Model Usage
history = credit.get_history()
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

# CreditScoring state
model_history = credit._history  # History of credit model calculations
```

## Next Steps

Now that you understand the basic components, explore these topics in detail:

- [Parray Guide](parray.md): Advanced technical analysis and signal detection
- [KPI Guide](kpi.md): Comprehensive business metrics and health scoring
- [Portfolio Guide](portfolio.md): Portfolio analysis and risk management
- [Service Pricing Guide](service-pricing.md): Pricing models and calculations
- [Credit Scoring Guide](credit-scoring.md): Credit risk assessment and scoring