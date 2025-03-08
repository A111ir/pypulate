---
search:
  boost: 2
features:
  - navigation.instant: false
---

# Expected Credit Loss (ECL)

The `expected_credit_loss` function calculates the Expected Credit Loss (ECL), a critical metric in credit risk management that estimates the probability-weighted loss on a financial asset. This metric is widely used in banking, lending, and financial risk management, especially since the introduction of IFRS 9 and similar accounting standards.

## Usage in Pypulate

```python
from pypulate.credit import expected_credit_loss

# Calculate ECL
result = expected_credit_loss(
    pd=0.05,           # 5% probability of default
    lgd=0.4,           # 40% loss given default
    ead=1000000,       # $1,000,000 exposure at default
    time_horizon=1.0,  # 1 year time horizon
    discount_rate=0.03 # 3% discount rate
)

# Access the results
ecl = result["expected_credit_loss"]
lifetime_ecl = result["lifetime_ecl"]
risk_level = result["risk_level"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `pd` | float | Probability of default (between 0 and 1) | Required |
| `lgd` | float | Loss given default (between 0 and 1) | Required |
| `ead` | float | Exposure at default | Required |
| `time_horizon` | float | Time horizon in years | 1.0 |
| `discount_rate` | float | Discount rate for future losses | 0.0 |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `expected_credit_loss` | float | The calculated ECL value |
| `lifetime_ecl` | float | The lifetime expected credit loss |
| `expected_loss_rate` | float | The product of PD and LGD |
| `risk_level` | str | Risk level categorization ("Very Low", "Low", "Moderate", "High", or "Very High") |
| `components` | dict | Dictionary containing calculation components |

The `components` dictionary includes:

- `pd`: Probability of default
- `lgd`: Loss given default
- `ead`: Exposure at default
- `time_horizon`: Time horizon in years
- `discount_rate`: Discount rate for future losses
- `discount_factor`: Calculated discount factor based on time horizon and discount rate

## Risk Level Classification

The Expected Loss Rate (PD × LGD) is categorized into risk levels:

| Expected Loss Rate Range | Risk Level |
|--------------------------|------------|
| < 1% | Very Low |
| 1% - 3% | Low |
| 3% - 7% | Moderate |
| 7% - 15% | High |
| > 15% | Very High |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze ECL for different loan portfolios:

```python
from pypulate.credit import expected_credit_loss
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Low-risk corporate loan
low_risk_loan = expected_credit_loss(
    pd=0.01,           # 1% probability of default
    lgd=0.3,           # 30% loss given default
    ead=2000000,       # $2,000,000 exposure
    time_horizon=1.0,  # 1 year
    discount_rate=0.02 # 2% discount rate
)

# Example 2: Medium-risk SME loan
medium_risk_loan = expected_credit_loss(
    pd=0.05,           # 5% probability of default
    lgd=0.45,          # 45% loss given default
    ead=500000,        # $500,000 exposure
    time_horizon=1.0,  # 1 year
    discount_rate=0.02 # 2% discount rate
)

# Example 3: High-risk unsecured consumer loan
high_risk_loan = expected_credit_loss(
    pd=0.15,           # 15% probability of default
    lgd=0.65,          # 65% loss given default
    ead=50000,         # $50,000 exposure
    time_horizon=1.0,  # 1 year
    discount_rate=0.02 # 2% discount rate
)

# Print the results
print("Expected Credit Loss Analysis")
print("============================")

print("\nExample 1: Low-risk Corporate Loan")
print(f"ECL: ${low_risk_loan['expected_credit_loss']:.2f}")
print(f"Expected Loss Rate: {low_risk_loan['expected_loss_rate']:.2%}")
print(f"Risk Level: {low_risk_loan['risk_level']}")

print("\nExample 2: Medium-risk SME Loan")
print(f"ECL: ${medium_risk_loan['expected_credit_loss']:.2f}")
print(f"Expected Loss Rate: {medium_risk_loan['expected_loss_rate']:.2%}")
print(f"Risk Level: {medium_risk_loan['risk_level']}")

print("\nExample 3: High-risk Consumer Loan")
print(f"ECL: ${high_risk_loan['expected_credit_loss']:.2f}")
print(f"Expected Loss Rate: {high_risk_loan['expected_loss_rate']:.2%}")
print(f"Risk Level: {high_risk_loan['risk_level']}")

# Create a DataFrame for visualization
loan_types = ['Corporate Loan', 'SME Loan', 'Consumer Loan']
ecl_values = [
    low_risk_loan['expected_credit_loss'],
    medium_risk_loan['expected_credit_loss'],
    high_risk_loan['expected_credit_loss']
]
loss_rates = [
    low_risk_loan['expected_loss_rate'],
    medium_risk_loan['expected_loss_rate'],
    high_risk_loan['expected_loss_rate']
]
exposures = [2000000, 500000, 50000]
risk_levels = [
    low_risk_loan['risk_level'],
    medium_risk_loan['risk_level'],
    high_risk_loan['risk_level']
]

# Create a bar chart for ECL comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(loan_types, ecl_values, color=['green', 'orange', 'red'])

# Add labels and title
plt.ylabel('Expected Credit Loss ($)')
plt.title('Expected Credit Loss Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add the ECL values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05 * max(ecl_values),
             f'${height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Create a scatter plot showing the relationship between exposure and ECL
plt.figure(figsize=(12, 6))

# Create a scatter plot with size proportional to exposure
sizes = [exposure/5000 for exposure in exposures]  # Scale for better visualization
colors = ['green', 'orange', 'red']

for i, loan_type in enumerate(loan_types):
    plt.scatter(loss_rates[i], ecl_values[i], s=sizes[i], 
                color=colors[i], alpha=0.7, label=f"{loan_type} (EAD: ${exposures[i]:,})")

# Add labels and title
plt.xlabel('Expected Loss Rate (PD × LGD)')
plt.ylabel('Expected Credit Loss ($)')
plt.title('ECL vs. Expected Loss Rate (Size represents Exposure at Default)')
plt.grid(True, linestyle='--', alpha=0.7)

# Add risk level regions
plt.axvspan(0, 0.01, alpha=0.2, color='green', label='Very Low Risk')
plt.axvspan(0.01, 0.03, alpha=0.2, color='lightgreen', label='Low Risk')
plt.axvspan(0.03, 0.07, alpha=0.2, color='yellow', label='Moderate Risk')
plt.axvspan(0.07, 0.15, alpha=0.2, color='orange', label='High Risk')
plt.axvspan(0.15, 1, alpha=0.2, color='red', label='Very High Risk')

plt.legend(loc='right', bbox_to_anchor=(1.25, 0.5), fontsize=9)
plt.tight_layout()
plt.show()

# Create a sensitivity analysis for PD and LGD
pd_values = np.linspace(0.01, 0.2, 20)
lgd_values = np.linspace(0.1, 0.9, 20)
PD, LGD = np.meshgrid(pd_values, lgd_values)
ELR = PD * LGD  # Expected Loss Rate

# Create risk level categories
risk_levels = np.zeros_like(ELR, dtype=str)
risk_levels = np.where(ELR < 0.01, 'Very Low', risk_levels)
risk_levels = np.where((ELR >= 0.01) & (ELR < 0.03), 'Low', risk_levels)
risk_levels = np.where((ELR >= 0.03) & (ELR < 0.07), 'Moderate', risk_levels)
risk_levels = np.where((ELR >= 0.07) & (ELR < 0.15), 'High', risk_levels)
risk_levels = np.where(ELR >= 0.15, 'Very High', risk_levels)

# Create a heatmap
plt.figure(figsize=(12, 8))
contour = plt.contourf(PD, LGD, ELR, levels=20, cmap='RdYlGn_r')
plt.colorbar(contour, label='Expected Loss Rate (PD × LGD)')

# Add contour lines for risk level boundaries
plt.contour(PD, LGD, ELR, levels=[0.01, 0.03, 0.07, 0.15], colors='black', linestyles='dashed')

# Add labels for risk regions
plt.text(0.005, 0.5, 'Very Low Risk', rotation=90, va='center', ha='center', color='black', fontweight='bold')
plt.text(0.02, 0.5, 'Low Risk', rotation=90, va='center', ha='center', color='black', fontweight='bold')
plt.text(0.05, 0.5, 'Moderate Risk', rotation=90, va='center', ha='center', color='black', fontweight='bold')
plt.text(0.11, 0.5, 'High Risk', rotation=90, va='center', ha='center', color='black', fontweight='bold')
plt.text(0.175, 0.5, 'Very High Risk', rotation=90, va='center', ha='center', color='black', fontweight='bold')

# Add labels and title
plt.xlabel('Probability of Default (PD)')
plt.ylabel('Loss Given Default (LGD)')
plt.title('Risk Level Sensitivity to PD and LGD')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
```

## Example Output

```
Expected Credit Loss Analysis
============================

Example 1: Low-risk Corporate Loan
ECL: $5882.35
Expected Loss Rate: 0.30%
Risk Level: Very Low

Example 2: Medium-risk SME Loan
ECL: $11029.41
Expected Loss Rate: 2.25%
Risk Level: Low

Example 3: High-risk Consumer Loan
ECL: $4779.41
Expected Loss Rate: 9.75%
Risk Level: High
```

## Visualizations

### ECL Comparison

This visualization compares the Expected Credit Loss for three different loan types, showing how the combination of risk factors and exposure amounts affects the total expected loss.

### ECL vs. Expected Loss Rate

This scatter plot shows the relationship between the Expected Loss Rate (PD × LGD) and the resulting ECL, with the size of each point representing the Exposure at Default. The background is color-coded to indicate different risk level regions.

![ECL Sensitivity](./charts/ecl_loss_rate.png)


### Risk Level Sensitivity

This heatmap demonstrates how the risk level changes with different combinations of PD and LGD, helping to visualize the sensitivity of the Expected Loss Rate to these two key parameters.

![ECL Sensitivity](./charts/ecl_sensitivity.png)

## Practical Applications

The Expected Credit Loss can be used for:

1. **IFRS 9 / CECL Compliance**: Meeting accounting standards for loan loss provisioning
2. **Credit Risk Management**: Quantifying and managing credit risk in loan portfolios
3. **Loan Pricing**: Incorporating expected losses into loan pricing models
4. **Capital Allocation**: Determining economic capital requirements for credit risk
5. **Portfolio Management**: Optimizing the risk-return profile of loan portfolios
6. **Stress Testing**: Assessing the impact of adverse economic scenarios on credit losses
7. **Risk-Based Limits**: Setting exposure limits based on expected loss considerations

## Industry Standards

Different financial institutions and regulatory frameworks may use slightly different approaches:

1. **Banking (Basel Framework)**:
    - Uses PD, LGD, and EAD for regulatory capital calculations
    - Typically requires through-the-cycle PD estimates
    - Downturn LGD estimates for capital adequacy

2. **Accounting Standards**:
    - IFRS 9: Forward-looking, point-in-time estimates with multiple economic scenarios
    - CECL: Lifetime expected losses from origination
    - Both require consideration of past events, current conditions, and reasonable forecasts

3. **Credit Rating Agencies**:
    - Provide expected loss estimates based on historical default and recovery data
    - Often use through-the-cycle methodologies for stability

## Best Practices

1. **Data Quality**: Ensure high-quality historical data for PD and LGD estimation
2. **Forward-Looking Adjustments**: Incorporate macroeconomic forecasts into PD and LGD estimates
3. **Segmentation**: Group exposures with similar risk characteristics
4. **Multiple Scenarios**: Consider various economic scenarios and their probabilities
5. **Regular Validation**: Backtest and validate ECL models regularly
6. **Expert Judgment**: Complement quantitative models with expert judgment
7. **Documentation**: Maintain comprehensive documentation of methodologies and assumptions
