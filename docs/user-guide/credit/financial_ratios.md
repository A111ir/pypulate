# Financial Ratios

The `financial_ratios` function calculates key financial ratios used in credit assessment and provides an overall assessment of a company's financial health. These ratios are grouped into categories including liquidity, solvency, profitability, coverage, and efficiency.

## Usage in Pypulate

```python
from pypulate.credit import financial_ratios

# Calculate financial ratios
result = financial_ratios(
    current_assets=250000,         # $250,000 current assets
    current_liabilities=100000,    # $100,000 current liabilities
    total_assets=1000000,          # $1,000,000 total assets
    total_liabilities=400000,      # $400,000 total liabilities
    ebit=150000,                   # $150,000 earnings before interest and taxes
    interest_expense=30000,        # $30,000 interest expense
    net_income=100000,             # $100,000 net income
    total_equity=600000,           # $600,000 total equity
    sales=800000                   # $800,000 sales
)

# Access the results
liquidity = result["liquidity"]
solvency = result["solvency"]
profitability = result["profitability"]
coverage = result["coverage"]
efficiency = result["efficiency"]
overall = result["overall_assessment"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `current_assets` | float | The company's current assets | Required |
| `current_liabilities` | float | The company's current liabilities | Required |
| `total_assets` | float | The company's total assets | Required |
| `total_liabilities` | float | The company's total liabilities | Required |
| `ebit` | float | Earnings before interest and taxes | Required |
| `interest_expense` | float | Interest expense | Required |
| `net_income` | float | Net income | Required |
| `total_equity` | float | Total equity | Required |
| `sales` | float | Total sales | Required |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `liquidity` | dict | Dictionary containing liquidity ratios and assessment |
| `solvency` | dict | Dictionary containing solvency ratios and assessment |
| `profitability` | dict | Dictionary containing profitability ratios and assessment |
| `coverage` | dict | Dictionary containing coverage ratios and assessment |
| `efficiency` | dict | Dictionary containing efficiency ratios |
| `overall_assessment` | str | Overall assessment of financial health |

The `liquidity` dictionary includes:
- `current_ratio`: Current assets divided by current liabilities
- `assessment`: Assessment of liquidity ("Strong", "Adequate", or "Weak")

The `solvency` dictionary includes:
- `debt_ratio`: Total liabilities divided by total assets
- `debt_to_equity`: Total liabilities divided by total equity
- `assessment`: Assessment of solvency ("Strong", "Adequate", or "Weak")

The `profitability` dictionary includes:
- `return_on_assets`: Net income divided by total assets
- `return_on_equity`: Net income divided by total equity
- `assessment`: Assessment of profitability ("Strong", "Adequate", or "Weak")

The `coverage` dictionary includes:
- `interest_coverage`: EBIT divided by interest expense
- `assessment`: Assessment of coverage ("Strong", "Adequate", or "Weak")

The `efficiency` dictionary includes:
- `asset_turnover`: Sales divided by total assets
- `assessment`: Assessment of efficiency ("Strong", "Adequate", or "Weak")

## Risk Level Classification

The financial ratios are categorized into assessment levels:

| Ratio | Range | Assessment |
|-------|-------|------------|
| Current Ratio | < 1.0 | Weak |
| | 1.0 - 2.0 | Adequate |
| | > 2.0 | Strong |
| Debt Ratio | > 0.6 | Weak |
| | 0.4 - 0.6 | Adequate |
| | < 0.4 | Strong |
| Debt-to-Equity | > 1.5 | Weak |
| | 0.5 - 1.5 | Adequate |
| | < 0.5 | Strong |
| Return on Assets | < 0.02 | Weak |
| | 0.02 - 0.05 | Adequate |
| | > 0.05 | Strong |
| Return on Equity | < 0.05 | Weak |
| | 0.05 - 0.15 | Adequate |
| | > 0.15 | Strong |
| Interest Coverage | < 1.5 | Weak |
| | 1.5 - 3.0 | Adequate |
| | > 3.0 | Strong |
| Asset Turnover | < 0.5 | Weak |
| | 0.5 - 1.0 | Adequate |
| | > 1.0 | Strong |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze financial ratios for different companies:

```python
from pypulate.credit import financial_ratios
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Financially strong company
strong_company = financial_ratios(
    current_assets=500000,         # $500,000 current assets
    current_liabilities=200000,    # $200,000 current liabilities
    total_assets=2000000,          # $2,000,000 total assets
    total_liabilities=600000,      # $600,000 total liabilities
    ebit=400000,                   # $400,000 earnings before interest and taxes
    interest_expense=50000,        # $50,000 interest expense
    net_income=300000,             # $300,000 net income
    total_equity=1400000,          # $1,400,000 total equity
    sales=2500000                  # $2,500,000 sales
)

# Example 2: Company with adequate financial health
adequate_company = financial_ratios(
    current_assets=300000,         # $300,000 current assets
    current_liabilities=200000,    # $200,000 current liabilities
    total_assets=1500000,          # $1,500,000 total assets
    total_liabilities=750000,      # $750,000 total liabilities
    ebit=200000,                   # $200,000 earnings before interest and taxes
    interest_expense=80000,        # $80,000 interest expense
    net_income=100000,             # $100,000 net income
    total_equity=750000,           # $750,000 total equity
    sales=1200000                  # $1,200,000 sales
)

# Example 3: Financially weak company
weak_company = financial_ratios(
    current_assets=150000,         # $150,000 current assets
    current_liabilities=200000,    # $200,000 current liabilities
    total_assets=1000000,          # $1,000,000 total assets
    total_liabilities=700000,      # $700,000 total liabilities
    ebit=50000,                    # $50,000 earnings before interest and taxes
    interest_expense=60000,        # $60,000 interest expense
    net_income=20000,              # $20,000 net income
    total_equity=300000,           # $300,000 total equity
    sales=600000                   # $600,000 sales
)

# Print the results
print("Financial Ratios Analysis")
print("========================")

print("\nExample 1: Strong Company")
print(f"Current Ratio: {strong_company['liquidity']['current_ratio']:.2f} ({strong_company['liquidity']['assessment']})")
print(f"Debt Ratio: {strong_company['solvency']['debt_ratio']:.2f} ({strong_company['solvency']['assessment']})")
print(f"Return on Equity: {strong_company['profitability']['return_on_equity']:.2f} ({strong_company['profitability']['assessment']})")
print(f"Interest Coverage: {strong_company['coverage']['interest_coverage']:.2f} ({strong_company['coverage']['assessment']})")
print(f"Asset Turnover: {strong_company['efficiency']['asset_turnover']:.2f}")
print(f"Overall Assessment: {strong_company['overall_assessment']}")

print("\nExample 2: Adequate Company")
print(f"Current Ratio: {adequate_company['liquidity']['current_ratio']:.2f} ({adequate_company['liquidity']['assessment']})")
print(f"Debt Ratio: {adequate_company['solvency']['debt_ratio']:.2f} ({adequate_company['solvency']['assessment']})")
print(f"Return on Equity: {adequate_company['profitability']['return_on_equity']:.2f} ({adequate_company['profitability']['assessment']})")
print(f"Interest Coverage: {adequate_company['coverage']['interest_coverage']:.2f} ({adequate_company['coverage']['assessment']})")
print(f"Asset Turnover: {adequate_company['efficiency']['asset_turnover']:.2f}")
print(f"Overall Assessment: {adequate_company['overall_assessment']}")

print("\nExample 3: Weak Company")
print(f"Current Ratio: {weak_company['liquidity']['current_ratio']:.2f} ({weak_company['liquidity']['assessment']})")
print(f"Debt Ratio: {weak_company['solvency']['debt_ratio']:.2f} ({weak_company['solvency']['assessment']})")
print(f"Return on Equity: {weak_company['profitability']['return_on_equity']:.2f} ({weak_company['profitability']['assessment']})")
print(f"Interest Coverage: {weak_company['coverage']['interest_coverage']:.2f} ({weak_company['coverage']['assessment']})")
print(f"Asset Turnover: {weak_company['efficiency']['asset_turnover']:.2f}")
print(f"Overall Assessment: {weak_company['overall_assessment']}")

# Visualize the results
companies = ['Strong', 'Adequate', 'Weak']
current_ratios = [
    strong_company['liquidity']['current_ratio'],
    adequate_company['liquidity']['current_ratio'],
    weak_company['liquidity']['current_ratio']
]
debt_ratios = [
    strong_company['solvency']['debt_ratio'],
    adequate_company['solvency']['debt_ratio'],
    weak_company['solvency']['debt_ratio']
]
interest_coverages = [
    strong_company['coverage']['interest_coverage'],
    adequate_company['coverage']['interest_coverage'],
    weak_company['coverage']['interest_coverage']
]
return_on_equities = [
    strong_company['profitability']['return_on_equity'],
    adequate_company['profitability']['return_on_equity'],
    weak_company['profitability']['return_on_equity']
]

# Create a comparison chart
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Financial Ratios Comparison', fontsize=16)

# Current Ratio
axs[0, 0].bar(companies, current_ratios, color=['green', 'orange', 'red'])
axs[0, 0].axhline(y=2, color='g', linestyle='--', label='Strong (≥ 2)')
axs[0, 0].axhline(y=1, color='r', linestyle='--', label='Weak (< 1)')
axs[0, 0].set_title('Current Ratio')
axs[0, 0].set_ylabel('Ratio')
axs[0, 0].legend()

# Debt Ratio
axs[0, 1].bar(companies, debt_ratios, color=['green', 'orange', 'red'])
axs[0, 1].axhline(y=0.4, color='g', linestyle='--', label='Strong (≤ 0.4)')
axs[0, 1].axhline(y=0.6, color='r', linestyle='--', label='Weak (> 0.6)')
axs[0, 1].set_title('Debt Ratio')
axs[0, 1].set_ylabel('Ratio')
axs[0, 1].legend()

# Interest Coverage
axs[1, 0].bar(companies, interest_coverages, color=['green', 'orange', 'red'])
axs[1, 0].axhline(y=3, color='g', linestyle='--', label='Strong (≥ 3)')
axs[1, 0].axhline(y=1.5, color='r', linestyle='--', label='Weak (< 1.5)')
axs[1, 0].set_title('Interest Coverage Ratio')
axs[1, 0].set_ylabel('Ratio')
axs[1, 0].legend()

# Return on Equity
axs[1, 1].bar(companies, return_on_equities, color=['green', 'orange', 'red'])
axs[1, 1].axhline(y=0.15, color='g', linestyle='--', label='Strong (≥ 0.15)')
axs[1, 1].axhline(y=0.08, color='r', linestyle='--', label='Weak (< 0.08)')
axs[1, 1].set_title('Return on Equity')
axs[1, 1].set_ylabel('Ratio')
axs[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Create a sensitivity analysis for current ratio
current_assets_values = np.linspace(100000, 500000, 100)  # Range of current assets values
current_liabilities = 200000  # Fixed current liabilities

current_ratios = [ca / current_liabilities for ca in current_assets_values]
assessments = []

for ratio in current_ratios:
    if ratio >= 2:
        assessments.append("Strong")
    elif ratio >= 1:
        assessments.append("Adequate")
    else:
        assessments.append("Weak")

# Create a plot showing how current ratio changes with current assets
plt.figure(figsize=(12, 6))

# Plot current ratio curve
plt.plot(current_assets_values, current_ratios, 'b-', linewidth=2)

# Add colored regions for different assessments
weak_indices = [i for i, r in enumerate(assessments) if r == "Weak"]
adequate_indices = [i for i, r in enumerate(assessments) if r == "Adequate"]
strong_indices = [i for i, r in enumerate(assessments) if r == "Strong"]

if weak_indices:
    plt.fill_between(current_assets_values[min(weak_indices):max(weak_indices)+1], 
                     0, current_ratios[min(weak_indices):max(weak_indices)+1], 
                     color='red', alpha=0.3, label='Weak')
if adequate_indices:
    plt.fill_between(current_assets_values[min(adequate_indices):max(adequate_indices)+1], 
                     0, current_ratios[min(adequate_indices):max(adequate_indices)+1], 
                     color='orange', alpha=0.3, label='Adequate')
if strong_indices:
    plt.fill_between(current_assets_values[min(strong_indices):max(strong_indices)+1], 
                     0, current_ratios[min(strong_indices):max(strong_indices)+1], 
                     color='green', alpha=0.3, label='Strong')

# Add horizontal lines for the ratio thresholds
plt.axhline(y=1, color='r', linestyle='--')
plt.axhline(y=2, color='g', linestyle='--')

# Add labels and title
plt.xlabel('Current Assets ($)')
plt.ylabel('Current Ratio')
plt.title('Current Ratio Sensitivity to Current Assets (Fixed Current Liabilities: $200,000)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
```

## Example Output

```
Financial Ratios Analysis
========================

Example 1: Strong Company
Current Ratio: 2.50 (Strong)
Debt Ratio: 0.30 (Strong)
Return on Equity: 0.21 (Strong)
Interest Coverage: 8.00 (Strong)
Asset Turnover: 1.25
Overall Assessment: Strong financial position

Example 2: Adequate Company
Current Ratio: 1.50 (Adequate)
Debt Ratio: 0.50 (Adequate)
Return on Equity: 0.13 (Adequate)
Interest Coverage: 2.50 (Adequate)
Asset Turnover: 0.80
Overall Assessment: Adequate financial position

Example 3: Weak Company
Current Ratio: 0.75 (Weak)
Debt Ratio: 0.70 (Weak)
Return on Equity: 0.07 (Weak)
Interest Coverage: 0.83 (Weak)
Asset Turnover: 0.60
Overall Assessment: Weak financial position
```

## Visualizations

### Financial Ratios Comparison

The following visualization shows a comparison of key financial ratios across three different companies (strong, adequate, and weak):

![Financial Ratios Comparison](./charts/financial_ratios_comparison.png)

This chart displays four key financial ratios:
- **Current Ratio**: Shows liquidity with thresholds at 2.0 (strong) and 1.0 (weak)
- **Debt Ratio**: Shows solvency with thresholds at 0.4 (strong) and 0.6 (weak)
- **Interest Coverage Ratio**: Shows debt service ability with thresholds at 3.0 (strong) and 1.5 (weak)
- **Return on Equity**: Shows profitability with thresholds at 0.15 (strong) and 0.08 (weak)

### Ratio Sensitivity Analysis

The following visualization demonstrates how the current ratio changes as current assets increase, while keeping current liabilities constant:

![Ratio Sensitivity Analysis](./charts/ratio_sensitivity.png)

This sensitivity analysis shows:
- The blue line represents the current ratio as current assets increase
- The red region represents the "Weak" assessment zone (ratio < 1.0)
- The orange region represents the "Adequate" assessment zone (1.0 ≤ ratio < 2.0)
- The green region represents the "Strong" assessment zone (ratio ≥ 2.0)
- The horizontal dashed lines mark the threshold values at 1.0 and 2.0

## Ratio Categories and Thresholds

### 1. Liquidity Ratios
Measure a company's ability to pay short-term obligations.

- **Current Ratio** = Current Assets / Current Liabilities
  - **Strong**: ≥ 2.0
  - **Adequate**: 1.0 - 2.0
  - **Weak**: < 1.0

### 2. Solvency Ratios
Measure a company's ability to meet long-term obligations.

- **Debt Ratio** = Total Liabilities / Total Assets
  - **Strong**: ≤ 0.4
  - **Adequate**: 0.4 - 0.6
  - **Weak**: > 0.6

- **Debt-to-Equity Ratio** = Total Liabilities / Total Equity
  - Lower values indicate better solvency

### 3. Profitability Ratios
Measure a company's ability to generate earnings relative to its assets and equity.

- **Return on Assets (ROA)** = Net Income / Total Assets
  - Higher values indicate better profitability

- **Return on Equity (ROE)** = Net Income / Total Equity
  - **Strong**: ≥ 0.15 (15%)
  - **Adequate**: 0.08 - 0.15 (8% - 15%)
  - **Weak**: < 0.08 (8%)

### 4. Coverage Ratios
Measure a company's ability to service its debt.

- **Interest Coverage Ratio** = EBIT / Interest Expense
  - **Strong**: ≥ 3.0
  - **Adequate**: 1.5 - 3.0
  - **Weak**: < 1.5

### 5. Efficiency Ratios
Measure how effectively a company uses its assets.

- **Asset Turnover Ratio** = Sales / Total Assets
  - Higher values indicate better efficiency

## Practical Applications

Financial ratios can be used for:

1. **Credit Risk Assessment**: Evaluating a borrower's financial health
2. **Investment Analysis**: Identifying financially stable companies
3. **Benchmarking**: Comparing a company's performance against industry peers
4. **Trend Analysis**: Monitoring changes in a company's financial health over time
5. **Covenant Compliance**: Ensuring borrowers maintain acceptable financial metrics

## Limitations

When using financial ratios, consider these limitations:

1. Industry differences may affect appropriate ratio values
2. Seasonal variations can impact short-term ratios
3. Accounting methods can affect ratio calculations
4. Historical ratios may not predict future performance
5. Ratios should be used alongside other financial metrics for comprehensive analysis 