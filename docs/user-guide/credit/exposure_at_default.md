# Exposure at Default (EAD)

The `exposure_at_default` function calculates the Exposure at Default (EAD), a key metric in credit risk management that estimates the expected amount outstanding when a borrower defaults. This metric is essential for credit risk modeling, regulatory capital calculations, and expected credit loss estimation under frameworks like Basel and IFRS 9.

## Usage in Pypulate

```python
from pypulate.credit import exposure_at_default

# Calculate EAD
result = exposure_at_default(
    current_balance=500000,    # $500,000 drawn amount
    undrawn_amount=500000,     # $500,000 undrawn commitment
    credit_conversion_factor=0.5  # 50% CCF
)

# Access the results
ead = result["ead"]
regulatory_ead = result["regulatory_ead"]
stressed_ead = result["stressed_ead"]
risk_level = result["risk_level"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `current_balance` | float | Current drawn amount of the credit facility | Required |
| `undrawn_amount` | float | Undrawn commitment available to the borrower | Required |
| `credit_conversion_factor` | float | Factor to convert undrawn amounts to exposure (between 0 and 1) | 0.5 (50%) |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `ead` | float | Exposure at Default using the provided CCF |
| `regulatory_ead` | float | EAD calculated using regulatory CCF based on utilization rate |
| `stressed_ead` | float | EAD calculated using a stressed CCF (1.5x the provided CCF, capped at 1.0) |
| `ead_percentage` | float | EAD as a percentage of total facility |
| `risk_level` | str | Risk level categorization ("Low", "Moderate", or "High") |
| `components` | dict | Dictionary containing calculation components |

The `components` dictionary includes:

- `current_balance`: The provided current balance
- `undrawn_amount`: The provided undrawn amount
- `total_facility`: Sum of current balance and undrawn amount
- `utilization_rate`: Current balance divided by total facility
- `credit_conversion_factor`: The provided CCF
- `regulatory_ccf`: CCF based on regulatory guidelines
- `stress_ccf`: Stressed CCF for scenario analysis

## Risk Level Classification

The Utilization Rate is categorized into risk levels:

| Utilization Rate Range | Risk Level |
|------------------------|------------|
| < 30% | Low |
| 30% - 70% | Moderate |
| > 70% | High |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze EAD for different credit facilities:

```python
from pypulate.credit import exposure_at_default
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Low utilization corporate credit line
low_util_facility = exposure_at_default(
    current_balance=200000,    # $200,000 drawn
    undrawn_amount=800000,     # $800,000 undrawn
    credit_conversion_factor=0.5  # 50% CCF
)

# Example 2: Medium utilization SME credit line
medium_util_facility = exposure_at_default(
    current_balance=500000,    # $500,000 drawn
    undrawn_amount=500000,     # $500,000 undrawn
    credit_conversion_factor=0.5  # 50% CCF
)

# Example 3: High utilization retail credit card
high_util_facility = exposure_at_default(
    current_balance=45000,     # $45,000 drawn
    undrawn_amount=5000,       # $5,000 undrawn
    credit_conversion_factor=0.5  # 50% CCF
)

# Print the results
print("Exposure at Default Analysis")
print("============================")

print("\nExample 1: Low Utilization Corporate Credit Line")
print(f"EAD: ${low_util_facility['ead']:.2f}")
print(f"Regulatory EAD: ${low_util_facility['regulatory_ead']:.2f}")
print(f"Stressed EAD: ${low_util_facility['stressed_ead']:.2f}")
print(f"Utilization Rate: {low_util_facility['components']['utilization_rate']:.2%}")
print(f"Risk Level: {low_util_facility['risk_level']}")

print("\nExample 2: Medium Utilization SME Credit Line")
print(f"EAD: ${medium_util_facility['ead']:.2f}")
print(f"Regulatory EAD: ${medium_util_facility['regulatory_ead']:.2f}")
print(f"Stressed EAD: ${medium_util_facility['stressed_ead']:.2f}")
print(f"Utilization Rate: {medium_util_facility['components']['utilization_rate']:.2%}")
print(f"Risk Level: {medium_util_facility['risk_level']}")

print("\nExample 3: High Utilization Retail Credit Card")
print(f"EAD: ${high_util_facility['ead']:.2f}")
print(f"Regulatory EAD: ${high_util_facility['regulatory_ead']:.2f}")
print(f"Stressed EAD: ${high_util_facility['stressed_ead']:.2f}")
print(f"Utilization Rate: {high_util_facility['components']['utilization_rate']:.2%}")
print(f"Risk Level: {high_util_facility['risk_level']}")

# Create a DataFrame for visualization
facility_types = ['Corporate Credit Line', 'SME Credit Line', 'Retail Credit Card']
ead_values = [
    low_util_facility['ead'],
    medium_util_facility['ead'],
    high_util_facility['ead']
]
regulatory_ead_values = [
    low_util_facility['regulatory_ead'],
    medium_util_facility['regulatory_ead'],
    high_util_facility['regulatory_ead']
]
stressed_ead_values = [
    low_util_facility['stressed_ead'],
    medium_util_facility['stressed_ead'],
    high_util_facility['stressed_ead']
]
utilization_rates = [
    low_util_facility['components']['utilization_rate'],
    medium_util_facility['components']['utilization_rate'],
    high_util_facility['components']['utilization_rate']
]
total_facilities = [
    low_util_facility['components']['total_facility'],
    medium_util_facility['components']['total_facility'],
    high_util_facility['components']['total_facility']
]

# Create a bar chart for EAD comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(facility_types))
width = 0.25

plt.bar(x - width, ead_values, width, label='Standard EAD', color='blue')
plt.bar(x, regulatory_ead_values, width, label='Regulatory EAD', color='green')
plt.bar(x + width, stressed_ead_values, width, label='Stressed EAD', color='red')

plt.xlabel('Facility Type')
plt.ylabel('Exposure at Default ($)')
plt.title('EAD Comparison Across Different Facility Types')
plt.xticks(x, facility_types)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add the utilization rate on top of each group
for i, rate in enumerate(utilization_rates):
    plt.text(i, max(ead_values[i], regulatory_ead_values[i], stressed_ead_values[i]) + 20000, 
             f'Utilization: {rate:.1%}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Create a scatter plot showing the relationship between utilization rate and EAD percentage
plt.figure(figsize=(12, 6))

# Create a scatter plot with size proportional to total facility
sizes = [facility/10000 for facility in total_facilities]  # Scale for better visualization
colors = ['green', 'orange', 'red']

for i, facility_type in enumerate(facility_types):
    plt.scatter(utilization_rates[i], 
                ead_values[i]/total_facilities[i], 
                s=sizes[i], 
                color=colors[i], 
                alpha=0.7, 
                label=f"{facility_type} (Total: ${total_facilities[i]:,})")

# Add labels and title
plt.xlabel('Utilization Rate')
plt.ylabel('EAD as % of Total Facility')
plt.title('EAD Percentage vs. Utilization Rate (Size represents Total Facility)')
plt.grid(True, linestyle='--', alpha=0.7)

# Add risk level regions
plt.axvspan(0, 0.3, alpha=0.2, color='green', label='Low Risk')
plt.axvspan(0.3, 0.7, alpha=0.2, color='yellow', label='Moderate Risk')
plt.axvspan(0.7, 1, alpha=0.2, color='red', label='High Risk')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Create a sensitivity analysis for utilization rate and CCF
utilization_values = np.linspace(0.1, 0.9, 9)
ccf_values = np.linspace(0.1, 1.0, 10)

# Create a matrix to store EAD percentages
ead_percentages = np.zeros((len(utilization_values), len(ccf_values)))

# Calculate EAD percentage for each combination
for i, util in enumerate(utilization_values):
    for j, ccf in enumerate(ccf_values):
        # For a total facility of 1.0, calculate EAD percentage
        current_balance = util * 1.0
        undrawn_amount = (1.0 - util)
        ead = current_balance + (undrawn_amount * ccf)
        ead_percentages[i, j] = ead

# Create a heatmap
plt.figure(figsize=(12, 8))
contour = plt.contourf(ccf_values, utilization_values, ead_percentages, levels=20, cmap='viridis')
plt.colorbar(contour, label='EAD as % of Total Facility')

# Add contour lines
plt.contour(ccf_values, utilization_values, ead_percentages, levels=10, colors='white', linestyles='dashed', linewidths=0.5)

# Add labels and title
plt.xlabel('Credit Conversion Factor (CCF)')
plt.ylabel('Utilization Rate')
plt.title('EAD Sensitivity to Utilization Rate and CCF')
plt.grid(True, linestyle='--', alpha=0.3)

# Add risk level indicators
plt.axhspan(0, 0.3, alpha=0.1, color='green', label='Low Risk')
plt.axhspan(0.3, 0.7, alpha=0.1, color='yellow', label='Moderate Risk')
plt.axhspan(0.7, 1, alpha=0.1, color='red', label='High Risk')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```

## Example Output

```
Exposure at Default Analysis
============================

Example 1: Low Utilization Corporate Credit Line
EAD: $600000.00
Regulatory EAD: $360000.00
Stressed EAD: $800000.00
Utilization Rate: 20.00%
Risk Level: Low

Example 2: Medium Utilization SME Credit Line
EAD: $750000.00
Regulatory EAD: $800000.00
Stressed EAD: $875000.00
Utilization Rate: 50.00%
Risk Level: Moderate

Example 3: High Utilization Retail Credit Card
EAD: $47500.00
Regulatory EAD: $49000.00
Stressed EAD: $48750.00
Utilization Rate: 90.00%
Risk Level: High
```

## Visualizations

### EAD Comparison

This visualization compares the standard EAD, regulatory EAD, and stressed EAD for three different facility types, showing how the utilization rate affects the exposure calculations.

![EAD Comparison](./charts/ead_comparison.png)

### EAD Percentage vs. Utilization Rate

This scatter plot shows the relationship between the utilization rate and the EAD as a percentage of the total facility, with the size of each point representing the total facility amount. The background is color-coded to indicate different risk level regions.

![EAD Risk Exposure](./charts/ead_risk_exposure.png)

### EAD Sensitivity

This heatmap demonstrates how the EAD percentage changes with different combinations of utilization rate and credit conversion factor, helping to visualize the sensitivity of the exposure calculation to these two key parameters.

![EAD Sensitivity](./charts/ead_sensitivity.png)

## Practical Applications

Exposure at Default calculations can be used for:

1. **Regulatory Capital**: Calculating regulatory capital requirements under Basel frameworks
2. **IFRS 9 / CECL**: Determining exposure inputs for expected credit loss calculations
3. **Credit Risk Management**: Quantifying potential exposure in credit facilities
4. **Limit Setting**: Establishing appropriate credit limits for different facility types
5. **Stress Testing**: Assessing the impact of increased drawdowns during stress scenarios
6. **Portfolio Management**: Understanding the risk profile of credit portfolios
7. **Pricing**: Incorporating potential exposure into risk-based pricing models

## Industry Standards

Different regulatory frameworks provide guidance on EAD calculation:

1. **Basel Framework**:
    - Standardized Approach: Prescribes fixed CCFs based on facility type
    - Internal Ratings-Based Approach: Allows banks to estimate their own CCFs
    - Typically differentiates between committed and uncommitted facilities

2. **Accounting Standards**:
    - IFRS 9: Requires consideration of expected drawdowns over the lifetime of the facility
    - CECL: Similar approach, focusing on lifetime exposure estimates

3. **Industry Practice**:
    - CCFs typically range from 0% (for uncommitted facilities) to 100% (for fully committed facilities)
    - Higher CCFs are applied to facilities with longer tenors and fewer covenants
    - Retail products often have product-specific CCFs based on historical behavior

## Best Practices

1. **Historical Analysis**: Base CCF estimates on historical drawdown behavior
2. **Segmentation**: Group facilities with similar characteristics for CCF estimation
3. **Stress Scenarios**: Consider increased drawdowns during economic downturns
4. **Facility Characteristics**: Account for commitment type, covenants, and maturity
5. **Regular Monitoring**: Track utilization rates and update CCF estimates periodically
6. **Conservative Approach**: Apply higher CCFs for facilities with uncertain drawdown patterns
7. **Documentation**: Maintain comprehensive documentation of methodologies and assumptions
