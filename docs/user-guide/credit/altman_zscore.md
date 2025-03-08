# Altman Z-Score

The Altman Z-Score is a financial metric used to predict the probability of a company going bankrupt within the next two years. Developed by Edward I. Altman in 1968, it combines five financial ratios with weighted coefficients to produce a single score that indicates financial health.

## Usage in Pypulate

```python
from pypulate.credit import altman_z_score


# Calculate Altman Z-Score
result = altman_z_score(
    working_capital=120000000,        # Working capital
    retained_earnings=200000000,      # Retained earnings
    ebit=80000000,                    # Earnings before interest and taxes
    market_value_equity=500000000,    # Market value of equity
    sales=350000000,                  # Sales
    total_assets=400000000,           # Total assets
    total_liabilities=150000000       # Total liabilities
)

# Access the results
z_score = result['z_score']
risk_assessment = result['risk_assessment']
zone = result['zone']
components = result['components']
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `working_capital` | float | Working capital (current assets - current liabilities) | Required |
| `retained_earnings` | float | Retained earnings | Required |
| `ebit` | float | Earnings before interest and taxes | Required |
| `market_value_equity` | float | Market value of equity | Required |
| `sales` | float | Annual sales | Required |
| `total_assets` | float | Total assets | Required |
| `total_liabilities` | float | Total liabilities | Required |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `z_score` | float | The calculated Z-Score value |
| `risk_assessment` | str | Text description of the bankruptcy risk |
| `zone` | str | Classification zone ("Safe", "Grey", or "Distress") |
| `interpretation` | str | Same as risk_assessment (for compatibility) |
| `components` | dict | Dictionary containing the individual ratio components |

The `components` dictionary includes:

- `x1`: Working capital / Total assets
- `x2`: Retained earnings / Total assets
- `x3`: EBIT / Total assets
- `x4`: Market value of equity / Total liabilities
- `x5`: Sales / Total assets

## Risk Level Classification

The Z-Score is categorized into risk zones:

| Z-Score Range | Risk Zone |
|---------------|-----------|
| > 2.99 | Safe Zone |
| 1.81 - 2.99 | Grey Zone |
| < 1.81 | Distress Zone |

## Comprehensive Example

Here's a complete example analyzing three companies with different financial profiles:

```python
from pypulate.credit import altman_z_score
import matplotlib.pyplot as plt
import numpy as np


# Example 1: Financially healthy manufacturing company
healthy_company = altman_z_score(
    working_capital=120000000,        # $120M working capital
    retained_earnings=200000000,      # $200M retained earnings
    ebit=80000000,                    # $80M earnings before interest and taxes
    market_value_equity=500000000,    # $500M market value of equity
    sales=350000000,                  # $350M annual sales
    total_assets=400000000,           # $400M total assets
    total_liabilities=150000000       # $150M total liabilities
)

# Example 2: Company in the "grey zone"
grey_zone_company = altman_z_score(
    working_capital=30000000,         # $30M working capital
    retained_earnings=40000000,       # $40M retained earnings
    ebit=25000000,                    # $25M earnings before interest and taxes
    market_value_equity=120000000,    # $120M market value of equity
    sales=200000000,                  # $200M annual sales
    total_assets=250000000,           # $250M total assets
    total_liabilities=150000000       # $150M total liabilities
)

# Example 3: Financially distressed company
distressed_company = altman_z_score(
    working_capital=5000000,          # $5M working capital
    retained_earnings=-20000000,      # -$20M retained earnings (accumulated losses)
    ebit=-8000000,                    # -$8M earnings before interest and taxes (operating loss)
    market_value_equity=30000000,     # $30M market value of equity
    sales=100000000,                  # $100M annual sales
    total_assets=150000000,           # $150M total assets
    total_liabilities=120000000       # $120M total liabilities
)

# Print the results
print("Altman Z-Score Analysis")
print("======================")

print("\nExample 1: Financially Healthy Company")
print(f"Z-Score: {healthy_company['z_score']:.2f}")
print(f"Risk Assessment: {healthy_company['risk_assessment']}")
print(f"Zone: {healthy_company['zone']}")
print("Component Values:")
for component, value in healthy_company['components'].items():
    print(f"  {component}: {value:.4f}")

print("\nExample 2: Grey Zone Company")
print(f"Z-Score: {grey_zone_company['z_score']:.2f}")
print(f"Risk Assessment: {grey_zone_company['risk_assessment']}")
print(f"Zone: {grey_zone_company['zone']}")
print("Component Values:")
for component, value in grey_zone_company['components'].items():
    print(f"  {component}: {value:.4f}")

print("\nExample 3: Financially Distressed Company")
print(f"Z-Score: {distressed_company['z_score']:.2f}")
print(f"Risk Assessment: {distressed_company['risk_assessment']}")
print(f"Zone: {distressed_company['zone']}")
print("Component Values:")
for component, value in distressed_company['components'].items():
    print(f"  {component}: {value:.4f}")

# Visualize the results
companies = ['Healthy', 'Grey Zone', 'Distressed']
z_scores = [
    healthy_company['z_score'],
    grey_zone_company['z_score'],
    distressed_company['z_score']
]

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(companies, z_scores, color=['green', 'orange', 'red'])

# Add horizontal lines for the Z-score thresholds
plt.axhline(y=1.81, color='r', linestyle='--', label='Distress Zone (Z < 1.81)')
plt.axhline(y=2.99, color='g', linestyle='--', label='Safe Zone (Z > 2.99)')
plt.axhspan(1.81, 2.99, alpha=0.2, color='orange', label='Grey Zone (1.81 < Z < 2.99)')

# Add labels and title
plt.ylabel('Altman Z-Score')
plt.title('Altman Z-Score Comparison')
plt.ylim(bottom=0)  # Start y-axis at 0

# Add the Z-score values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.show()
```

## Example Output

```
Altman Z-Score Analysis
======================
Example 1: Financially Healthy Company
Z-Score: 4.59
Risk Assessment: Low risk of bankruptcy
Zone: Safe
Component Values:
  x1: 0.3000
  x2: 0.5000
  x3: 0.2000
  x4: 3.3333
  x5: 0.8750
Example 2: Grey Zone Company
Z-Score: 1.98
Risk Assessment: Grey area, moderate risk
Zone: Grey
Component Values:
  x1: 0.1200
  x2: 0.1600
  x3: 0.1000
  x4: 0.8000
  x5: 0.8000
Example 3: Financially Distressed Company
Z-Score: 0.49
Risk Assessment: High risk of bankruptcy
Zone: Distress
Component Values:
  x1: 0.0333
  x2: -0.1333
  x3: -0.0533
  x4: 0.2500
  x5: 0.6667
```

## Visualization

![Altman Z-Score Comparison](./charts/altmanzscore.png)

The visualization shows the Z-scores for three example companies, with horizontal lines indicating the threshold values that separate the Safe, Grey, and Distress zones.

## Component Analysis

Each component of the Z-Score provides insight into different aspects of a company's financial health:

1. **X₁ (Working Capital / Total Assets)**
    - Measures liquidity relative to company size
    - Higher values indicate better short-term financial health
    - Weight: 1.2

2. **X₂ (Retained Earnings / Total Assets)**
    - Measures cumulative profitability and company age
    - Higher values indicate stronger historical performance
    - Weight: 1.4

3. **X₃ (EBIT / Total Assets)**
    - Measures operating efficiency independent of tax and leverage
    - Has the highest weight (3.3), indicating its importance
    - Higher values indicate better operational performance

4. **X₄ (Market Value of Equity / Total Liabilities)**
    - Measures financial leverage and market confidence
    - Higher values indicate lower financial risk
    - Weight: 0.6

5. **X₅ (Sales / Total Assets)**
    - Measures asset turnover and management efficiency
    - Higher values indicate better utilization of assets
    - Weight: 0.999

## Practical Applications

The Altman Z-Score can be used for:

1. **Credit Risk Assessment**: Evaluating potential borrowers' bankruptcy risk
2. **Investment Screening**: Identifying financially stable companies
3. **Portfolio Risk Management**: Monitoring existing investments
4. **Supplier Evaluation**: Assessing the financial stability of key suppliers
5. **Merger & Acquisition Analysis**: Evaluating target companies

## Limitations

While the Altman Z-Score is a powerful tool, it has some limitations:

1. Originally developed for manufacturing companies
2. May need industry-specific adjustments
3. Works best for public companies with market value data
4. Should be used alongside other financial metrics
5. Historical performance may not predict future outcomes
