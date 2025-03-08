# Credit Scoring and Risk Assessment

This document demonstrates the credit scoring and risk assessment capabilities available in the `pypulate` library using the `CreditScoring` class.

## Overview

The `CreditScoring` class provides a comprehensive suite of credit risk models and assessment tools:

1. Altman Z-Score for bankruptcy prediction
2. Merton Model for default probability
3. Debt Service Coverage Ratio (DSCR)
4. Weight of Evidence (WOE) and Information Value (IV)
5. Logistic Regression Scoring
6. Credit Scorecard Creation
7. Credit Rating Transition Matrix
8. Financial Ratios Analysis
9. Expected Credit Loss (ECL) Calculation
10. Risk-Based Loan Pricing
11. Scoring Model Validation
12. Loss Given Default (LGD) Estimation
13. Exposure at Default (EAD) Calculation

## Basic Usage

First, let's import the necessary components:

```python
from pypulate import CreditScoring
```

## Corporate Credit Risk Assessment

Let's start with a basic corporate credit risk assessment using the Altman Z-Score model:

```python
# Initialize the CreditScoring class
credit = CreditScoring()

# Company financial data
working_capital = 1200000
retained_earnings = 1500000
ebit = 800000
market_value_equity = 5000000
sales = 4500000
total_assets = 6000000
total_liabilities = 2500000

# Calculate Altman Z-Score
z_score_result = credit.altman_z_score(
    working_capital=working_capital,
    retained_earnings=retained_earnings,
    ebit=ebit,
    market_value_equity=market_value_equity,
    sales=sales,
    total_assets=total_assets,
    total_liabilities=total_liabilities
)

print(f"Altman Z-Score: {z_score_result['z_score']:.2f}")
print(f"Risk Assessment: {z_score_result['risk_assessment']}")
```

## Default Probability Estimation

The Merton model provides a structural approach to estimating default probability:

```python
# Company data for Merton model
asset_value = 10000000
debt_face_value = 5000000
asset_volatility = 0.25
risk_free_rate = 0.03
time_to_maturity = 1.0

# Calculate default probability using Merton model
merton_result = credit.merton_model(
    asset_value=asset_value,
    debt_face_value=debt_face_value,
    asset_volatility=asset_volatility,
    risk_free_rate=risk_free_rate,
    time_to_maturity=time_to_maturity
)

print(f"Probability of Default: {merton_result['probability_of_default']:.2%}")
print(f"Distance to Default: {merton_result['distance_to_default']:.2f}")
```

## Debt Servicing Capacity

The Debt Service Coverage Ratio helps assess a borrower's ability to service debt:

```python
# Debt service data
net_operating_income = 500000
total_debt_service = 300000

# Calculate DSCR
dscr_result = credit.debt_service_coverage_ratio(
    net_operating_income=net_operating_income,
    total_debt_service=total_debt_service
)

print(f"DSCR: {dscr_result['dscr']:.2f}")
print(f"Assessment: {dscr_result['assessment']}")
```

## Credit Scorecard Development

Create a points-based credit scorecard for retail lending:

```python
# Applicant features
features = {
    "age": 35,
    "income": 75000,
    "years_employed": 5,
    "debt_to_income": 0.3,
    "previous_defaults": 0
}

# Feature weights (derived from statistical analysis)
weights = {
    "age": 2.5,
    "income": 3.2,
    "years_employed": 4.0,
    "debt_to_income": -5.5,
    "previous_defaults": -25.0
}

# Feature offsets (reference points)
offsets = {
    "age": 25,
    "income": 50000,
    "years_employed": 2,
    "debt_to_income": 0.4,
    "previous_defaults": 1
}

# Create scorecard
scorecard_result = credit.create_scorecard(
    features=features,
    weights=weights,
    offsets=offsets,
    scaling_factor=20,
    base_score=600
)

print(f"Total Credit Score: {scorecard_result['total_score']:.0f}")
print(f"Risk Category: {scorecard_result['risk_category']}")
print("\nPoints Breakdown:")
for feature, points in scorecard_result['points_breakdown'].items():
    print(f"  {feature}: {points:.0f} points")
```

## Expected Credit Loss Calculation

Calculate the expected credit loss for a loan:

```python
# Loan risk parameters
pd = 0.05  # Probability of default
lgd = 0.4  # Loss given default
ead = 100000  # Exposure at default
time_horizon = 1.0  # 1 year
discount_rate = 0.03  # 3% discount rate

# Calculate ECL
ecl_result = credit.expected_credit_loss(
    pd=pd,
    lgd=lgd,
    ead=ead,
    time_horizon=time_horizon,
    discount_rate=discount_rate
)

print(f"Expected Credit Loss: ${ecl_result['ecl']:.2f}")
print(f"ECL as % of Exposure: {ecl_result['ecl_percentage']:.2%}")
```

## Risk-Based Loan Pricing

Determine the appropriate interest rate for a loan based on risk:

```python
# Loan and risk parameters
loan_amount = 250000
term = 5  # 5 years
pd = 0.03  # Annual probability of default
lgd = 0.35  # Loss given default
funding_cost = 0.04  # Cost of funds
operating_cost = 0.01  # Operating costs as % of loan
capital_requirement = 0.08  # Capital requirement
target_roe = 0.15  # Target return on equity

# Calculate loan pricing
pricing_result = credit.loan_pricing(
    loan_amount=loan_amount,
    term=term,
    pd=pd,
    lgd=lgd,
    funding_cost=funding_cost,
    operating_cost=operating_cost,
    capital_requirement=capital_requirement,
    target_roe=target_roe
)

print(f"Recommended Interest Rate: {pricing_result['recommended_rate']:.2%}")
print(f"Effective Annual Rate: {pricing_result['effective_annual_rate']:.2%}")
print("\nRate Components:")
for component, value in pricing_result['components'].items():
    print(f"  {component}: {value:.2%}")
```

## Credit Rating Transition Analysis

Analyze how credit ratings migrate over time:

```python
# Historical credit ratings data
ratings_t0 = ['AAA', 'AA', 'A', 'BBB', 'BB', 'A', 'BBB', 'BB', 'B', 'CCC']  # Initial ratings
ratings_t1 = ['AA', 'A', 'BBB', 'BB', 'B', 'A', 'BBB', 'CCC', 'CCC', 'D']   # Ratings after 1 year

# Calculate transition matrix
transition_result = credit.transition_matrix(
    ratings_t0=ratings_t0,
    ratings_t1=ratings_t1
)

print("Credit Rating Transition Matrix (Probabilities):")
prob_matrix = transition_result['probability_matrix']
ratings = transition_result['ratings']

# Print the matrix with proper formatting
print(f"{'':5}", end="")
for r in ratings:
    print(f"{r:6}", end="")
print()

for i, row in enumerate(prob_matrix):
    print(f"{ratings[i]:5}", end="")
    for val in row:
        print(f"{val:.2f}  ", end="")
    print()
```

## Financial Ratios Analysis

Analyze a company's financial health through key ratios:

```python
# Company financial data
current_assets = 2000000
current_liabilities = 1200000
total_assets = 8000000
total_liabilities = 4000000
ebit = 1200000
interest_expense = 300000
net_income = 700000
total_equity = 4000000
sales = 6000000

# Calculate financial ratios
ratios_result = credit.financial_ratios(
    current_assets=current_assets,
    current_liabilities=current_liabilities,
    total_assets=total_assets,
    total_liabilities=total_liabilities,
    ebit=ebit,
    interest_expense=interest_expense,
    net_income=net_income,
    total_equity=total_equity,
    sales=sales
)

print("Key Financial Ratios:")
for category, ratios in ratios_result.items():
    if category != 'overall_assessment':
        print(f"\n{category.replace('_', ' ').title()}:")
        for ratio_name, value in ratios.items():
            if ratio_name != 'assessment':
                print(f"  {ratio_name.replace('_', ' ').title()}: {value:.2f}")
        print(f"  Assessment: {ratios['assessment']}")

print(f"\nOverall Financial Health: {ratios_result['overall_assessment']}")
```

## Scoring Model Validation

Validate the performance of a credit scoring model:

```python
import numpy as np

# Simulated data: predicted scores and actual defaults
np.random.seed(42)
num_samples = 1000
predicted_scores = np.random.normal(650, 100, num_samples)
# Higher scores should correspond to lower default probability
default_probs = 1 / (1 + np.exp((predicted_scores - 600) / 50))
actual_defaults = np.random.binomial(1, default_probs)

# Validate the scoring model
validation_result = credit.scoring_model_validation(
    predicted_scores=predicted_scores,
    actual_defaults=actual_defaults,
    score_bins=10
)

print(f"Gini Coefficient: {validation_result['gini']:.4f}")
print(f"KS Statistic: {validation_result['ks']:.4f}")
print(f"AUC-ROC: {validation_result['auc']:.4f}")
print(f"Accuracy: {validation_result['accuracy']:.4f}")
```

## Loss Given Default Estimation

Estimate the loss given default for a secured loan:

```python
# Loan and collateral data
collateral_value = 180000
loan_amount = 200000
liquidation_costs = 0.15
time_to_recovery = 1.5

# Calculate LGD
lgd_result = credit.loss_given_default(
    collateral_value=collateral_value,
    loan_amount=loan_amount,
    liquidation_costs=liquidation_costs,
    time_to_recovery=time_to_recovery
)

print(f"Loss Given Default: {lgd_result['lgd']:.2%}")
print(f"Recovery Rate: {lgd_result['recovery_rate']:.2%}")
print(f"Expected Loss Amount: ${lgd_result['expected_loss']:.2f}")
```

## Exposure at Default Calculation

Calculate the exposure at default for a credit facility:

```python
# Credit facility data
current_balance = 500000
undrawn_amount = 300000
credit_conversion_factor = 0.6

# Calculate EAD
ead_result = credit.exposure_at_default(
    current_balance=current_balance,
    undrawn_amount=undrawn_amount,
    credit_conversion_factor=credit_conversion_factor
)

print(f"Exposure at Default: ${ead_result['ead']:.2f}")
print(f"EAD Components:")
print(f"  Drawn Balance: ${ead_result['drawn_balance']:.2f}")
print(f"  Expected Draw: ${ead_result['expected_draw']:.2f}")
```

## Tracking Model Usage

The CreditScoring class maintains a history of all calculations:

```python
# Get history of calculations
history = credit.get_history()

print(f"Number of calculations performed: {len(history)}")
print("\nRecent calculations:")
for i, entry in enumerate(history[-3:]):
    print(f"{i+1}. Model: {entry['model']}")
```

## Best Practices

### 1. Data Quality
  - Data Cleaning: Ensure financial data is accurate and complete
  - Outlier Treatment: Handle outliers appropriately
  - Missing Values: Develop a consistent approach for missing data

### 2. Model Selection
  - Purpose Fit: Choose models appropriate for the specific credit assessment need
  - Complexity: Balance model complexity with interpretability
  - Validation: Regularly validate model performance

### 3. Risk Management
  - Stress Testing: Test models under adverse scenarios
  - Sensitivity Analysis: Understand how changes in inputs affect outputs
  - Model Limitations: Be aware of each model's limitations

### 4. Implementation
  - Documentation: Document assumptions and methodologies
  - Monitoring: Regularly monitor model performance
  - Updating: Update models as economic conditions change

## Common Pitfalls

### 1. Model Misuse
  - Inappropriate Application: Using models outside their intended domain
  - Overreliance: Relying too heavily on quantitative models without qualitative assessment
  - Outdated Models: Using models that haven't been updated for current conditions

### 2. Data Issues
  - Sample Bias: Training on non-representative data
  - Look-ahead Bias: Using information not available at decision time
  - Data Staleness: Using outdated financial information

### 3. Implementation Challenges
  - Parameter Sensitivity: Results highly sensitive to input parameters
  - Model Risk: Risk of model error or misspecification
  - Interpretation Errors: Misinterpreting model outputs 