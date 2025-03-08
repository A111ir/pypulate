# Loan Pricing

The `loan_pricing` function implements a risk-based loan pricing model that calculates the appropriate interest rate for a loan based on various risk factors and cost components. This model helps lenders determine fair and profitable loan terms while accounting for the borrower's risk profile.

## Components of Loan Pricing

The loan pricing model considers several key components:

1. **Expected Loss Component**: Accounts for the probability of default and the expected loss given default
2. **Funding Cost Component**: Reflects the lender's cost of obtaining funds
3. **Operating Cost Component**: Covers the administrative costs of originating and servicing the loan
4. **Capital Cost Component**: Accounts for the required return on the capital allocated to support the loan

## Usage in Pypulate

```python
from pypulate.credit import loan_pricing

# Calculate risk-based loan pricing
result = loan_pricing(
    loan_amount=100000,           # $100,000 loan
    term=5,                       # 5-year term
    pd=0.02,                      # 2% annual probability of default
    lgd=0.4,                      # 40% loss given default
    funding_cost=0.03,            # 3% cost of funds
    operating_cost=0.01,          # 1% operating costs
    capital_requirement=0.08,     # 8% capital requirement
    target_roe=0.15               # 15% target return on equity
)

# Access the results
recommended_rate = result["recommended_rate"]
monthly_payment = result["monthly_payment"]
components = result["components"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `loan_amount` | float | The principal amount of the loan | Required |
| `term` | float | The loan term in years | Required |
| `pd` | float | Probability of default (annual rate, between 0 and 1) | Required |
| `lgd` | float | Loss given default (as a decimal, between 0 and 1) | Required |
| `funding_cost` | float | Cost of funds (annual rate) | Required |
| `operating_cost` | float | Operating costs as percentage of loan amount | Required |
| `capital_requirement` | float | Capital requirement as percentage of loan amount | Required |
| `target_roe` | float | Target return on equity (annual rate) | Required |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `recommended_rate` | float | The calculated interest rate for the loan |
| `effective_annual_rate` | float | The effective annual rate (APR) |
| `monthly_payment` | float | The calculated monthly payment amount |
| `total_interest` | float | Total interest paid over the life of the loan |
| `expected_profit` | float | Expected profit after accounting for losses and costs |
| `return_on_investment` | float | Expected return on the allocated capital |
| `components` | dict | Dictionary containing the individual pricing components |

The `components` dictionary includes:
- `expected_loss`: Component accounting for credit risk
- `funding_cost`: Component accounting for cost of funds
- `operating_cost`: Component accounting for operational expenses
- `capital_cost`: Component accounting for capital allocation
- `risk_premium`: Combined risk-related components (expected_loss + capital_cost)

## Risk Level Classification

The loan risk is categorized based on the expected loss rate (PD × LGD):

| Expected Loss Rate Range | Risk Level |
|--------------------------|------------|
| < 1% | Very Low |
| 1% - 3% | Low |
| 3% - 7% | Moderate |
| 7% - 15% | High |
| > 15% | Very High |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze loan pricing for different risk profiles:

```python
from pypulate.credit import loan_pricing
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Low-risk borrower
low_risk_loan = loan_pricing(
    loan_amount=100000,           # $100,000 loan
    term=5,                       # 5-year term
    pd=0.01,                      # 1% probability of default
    lgd=0.3,                      # 30% loss given default
    funding_cost=0.03,            # 3% cost of funds
    operating_cost=0.01,          # 1% operating costs
    capital_requirement=0.08,     # 8% capital requirement
    target_roe=0.15               # 15% target return on equity
)

# Example 2: Medium-risk borrower
medium_risk_loan = loan_pricing(
    loan_amount=100000,           # $100,000 loan
    term=5,                       # 5-year term
    pd=0.03,                      # 3% probability of default
    lgd=0.4,                      # 40% loss given default
    funding_cost=0.03,            # 3% cost of funds
    operating_cost=0.01,          # 1% operating costs
    capital_requirement=0.08,     # 8% capital requirement
    target_roe=0.15               # 15% target return on equity
)

# Example 3: High-risk borrower
high_risk_loan = loan_pricing(
    loan_amount=100000,           # $100,000 loan
    term=5,                       # 5-year term
    pd=0.08,                      # 8% probability of default
    lgd=0.5,                      # 50% loss given default
    funding_cost=0.03,            # 3% cost of funds
    operating_cost=0.01,          # 1% operating costs
    capital_requirement=0.08,     # 8% capital requirement
    target_roe=0.15               # 15% target return on equity
)

# Print the results
print("Risk-Based Loan Pricing Analysis")
print("===============================")

print("\nExample 1: Low-Risk Borrower")
print(f"Recommended Interest Rate: {low_risk_loan['recommended_rate']:.2%}")
print(f"Effective Annual Rate: {low_risk_loan['effective_annual_rate']:.2%}")
print(f"Monthly Payment: ${low_risk_loan['monthly_payment']:.2f}")
print(f"Total Interest: ${low_risk_loan['total_interest']:.2f}")
print(f"Expected Profit: ${low_risk_loan['expected_profit']:.2f}")
print(f"Return on Investment: {low_risk_loan['return_on_investment']:.2%}")
print("Pricing Components:")
for component, value in low_risk_loan['components'].items():
    print(f"  {component}: {value:.2%}")

print("\nExample 2: Medium-Risk Borrower")
print(f"Recommended Interest Rate: {medium_risk_loan['recommended_rate']:.2%}")
print(f"Effective Annual Rate: {medium_risk_loan['effective_annual_rate']:.2%}")
print(f"Monthly Payment: ${medium_risk_loan['monthly_payment']:.2f}")
print(f"Total Interest: ${medium_risk_loan['total_interest']:.2f}")
print(f"Expected Profit: ${medium_risk_loan['expected_profit']:.2f}")
print(f"Return on Investment: {medium_risk_loan['return_on_investment']:.2%}")
print("Pricing Components:")
for component, value in medium_risk_loan['components'].items():
    print(f"  {component}: {value:.2%}")

print("\nExample 3: High-Risk Borrower")
print(f"Recommended Interest Rate: {high_risk_loan['recommended_rate']:.2%}")
print(f"Effective Annual Rate: {high_risk_loan['effective_annual_rate']:.2%}")
print(f"Monthly Payment: ${high_risk_loan['monthly_payment']:.2f}")
print(f"Total Interest: ${high_risk_loan['total_interest']:.2f}")
print(f"Expected Profit: ${high_risk_loan['expected_profit']:.2f}")
print(f"Return on Investment: {high_risk_loan['return_on_investment']:.2%}")
print("Pricing Components:")
for component, value in high_risk_loan['components'].items():
    print(f"  {component}: {value:.2%}")

# Visualize the results - Interest Rate Comparison
risk_profiles = ['Low Risk', 'Medium Risk', 'High Risk']
interest_rates = [
    low_risk_loan['recommended_rate'],
    medium_risk_loan['recommended_rate'],
    high_risk_loan['recommended_rate']
]

plt.figure(figsize=(10, 6))
bars = plt.bar(risk_profiles, interest_rates, color=['green', 'orange', 'red'])

# Add the rate values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.2%}', ha='center', va='bottom')

plt.ylabel('Recommended Interest Rate')
plt.title('Risk-Based Interest Rate Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualize the pricing components
components = ['Expected Loss', 'Funding Cost', 'Operating Cost', 'Capital Cost']
low_risk_components = [
    low_risk_loan['components']['expected_loss'],
    low_risk_loan['components']['funding_cost'],
    low_risk_loan['components']['operating_cost'],
    low_risk_loan['components']['capital_cost']
]
medium_risk_components = [
    medium_risk_loan['components']['expected_loss'],
    medium_risk_loan['components']['funding_cost'],
    medium_risk_loan['components']['operating_cost'],
    medium_risk_loan['components']['capital_cost']
]
high_risk_components = [
    high_risk_loan['components']['expected_loss'],
    high_risk_loan['components']['funding_cost'],
    high_risk_loan['components']['operating_cost'],
    high_risk_loan['components']['capital_cost']
]

x = np.arange(len(components))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, low_risk_components, width, label='Low Risk', color='green')
rects2 = ax.bar(x, medium_risk_components, width, label='Medium Risk', color='orange')
rects3 = ax.bar(x + width, high_risk_components, width, label='High Risk', color='red')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Rate Component')
ax.set_title('Loan Pricing Components by Risk Profile')
ax.set_xticks(x)
ax.set_xticklabels(components)
ax.legend()

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Create a sensitivity analysis for PD
pd_values = np.linspace(0.01, 0.10, 10)  # Range of PD values from 1% to 10%
interest_rates = []
monthly_payments = []
expected_profits = []

for pd_value in pd_values:
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=pd_value,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    interest_rates.append(result['recommended_rate'])
    monthly_payments.append(result['monthly_payment'])
    expected_profits.append(result['expected_profit'])

# Plot interest rate sensitivity to PD
plt.figure(figsize=(12, 6))
plt.plot(pd_values * 100, np.array(interest_rates) * 100, 'b-', linewidth=2, marker='o')
plt.xlabel('Probability of Default (%)')
plt.ylabel('Recommended Interest Rate (%)')
plt.title('Interest Rate Sensitivity to Probability of Default')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot monthly payment sensitivity to PD
plt.figure(figsize=(12, 6))
plt.plot(pd_values * 100, monthly_payments, 'g-', linewidth=2, marker='o')
plt.xlabel('Probability of Default (%)')
plt.ylabel('Monthly Payment ($)')
plt.title('Monthly Payment Sensitivity to Probability of Default')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot expected profit sensitivity to PD
plt.figure(figsize=(12, 6))
plt.plot(pd_values * 100, expected_profits, 'r-', linewidth=2, marker='o')
plt.xlabel('Probability of Default (%)')
plt.ylabel('Expected Profit ($)')
plt.title('Expected Profit Sensitivity to Probability of Default')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

## Example Output

```
Risk-Based Loan Pricing Analysis
===============================

Example 1: Low-Risk Borrower
Recommended Interest Rate: 4.46%
Effective Annual Rate: 4.55%
Monthly Payment: $1862.48
Total Interest: $11749.02
Expected Profit: $6449.02
Return on Investment: 80.61%
Pricing Components:
  expected_loss: 0.06%
  funding_cost: 3.00%
  operating_cost: 0.20%
  capital_cost: 1.20%
  risk_premium: 1.26%

Example 2: Medium-Risk Borrower
Recommended Interest Rate: 4.64%
Effective Annual Rate: 4.74%
Monthly Payment: $1870.67
Total Interest: $12240.48
Expected Profit: $6040.48
Return on Investment: 75.51%
Pricing Components:
  expected_loss: 0.24%
  funding_cost: 3.00%
  operating_cost: 0.20%
  capital_cost: 1.20%
  risk_premium: 1.44%

Example 3: High-Risk Borrower
Recommended Interest Rate: 5.20%
Effective Annual Rate: 5.33%
Monthly Payment: $1896.30
Total Interest: $13778.00
Expected Profit: $4778.00
Return on Investment: 59.72%
Pricing Components:
  expected_loss: 0.80%
  funding_cost: 3.00%
  operating_cost: 0.20%
  capital_cost: 1.20%
  risk_premium: 2.00%
```

## Pricing Component Analysis

Each component of the loan pricing model serves a specific purpose:

1. **Expected Loss Component** (pd × lgd / term)
    - Compensates for the expected credit losses
    - Directly proportional to both probability of default and loss severity
    - Higher risk borrowers have significantly higher expected loss components

2. **Funding Cost Component**
    - Represents the lender's cost of obtaining the funds to lend
    - Typically based on market interest rates
    - Generally consistent across borrowers regardless of risk

3. **Operating Cost Component** (operating_cost / term)
    - Covers origination, servicing, and administrative costs
    - Spread over the life of the loan
    - May vary slightly based on loan complexity

4. **Capital Cost Component** (capital_requirement × target_roe)
    - Compensates for the opportunity cost of capital allocated to the loan
    - Higher risk loans may require more capital allocation
    - Reflects the lender's required return on invested capital

5. **Risk Premium** (expected_loss + capital_cost)
    - The combined risk-related components
    - Represents the additional return required to compensate for risk
    - Primary differentiator in pricing between low and high-risk borrowers

## Practical Applications

Risk-based loan pricing can be used for:

1. **Consumer Lending**: Setting appropriate rates for personal loans, auto loans, and mortgages
2. **Commercial Lending**: Pricing business loans based on company financial health
3. **Credit Card Pricing**: Determining APRs for different customer segments
4. **Loan Portfolio Management**: Ensuring adequate returns across a portfolio of loans
5. **Competitive Analysis**: Benchmarking pricing against market competitors

## Limitations and Considerations

When using risk-based loan pricing, consider these limitations:

1. **Model Assumptions**: The accuracy depends on reliable estimates of PD and LGD
2. **Market Constraints**: Competitive pressures may limit the ability to charge risk-appropriate rates
3. **Regulatory Considerations**: Fair lending laws may restrict risk-based pricing in some markets
4. **Customer Acceptance**: Very high rates may lead to adverse selection or reduced demand
5. **Economic Cycles**: Risk parameters should be adjusted for changing economic conditions 