# Debt Service Coverage Ratio (DSCR)

The `debt_service_coverage_ratio` function calculates the Debt Service Coverage Ratio (DSCR), a key financial metric used to assess a borrower's ability to repay debt. This ratio is widely used in commercial real estate lending, corporate finance, and credit risk assessment.

## Usage in Pypulate

```python
from pypulate.credit import debt_service_coverage_ratio

# Calculate DSCR
result = debt_service_coverage_ratio(
    net_operating_income=500000,  # $500,000 net operating income
    total_debt_service=300000     # $300,000 total debt service
)

# Access the results
dscr = result["dscr"]
assessment = result["assessment"]
rating = result["rating"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `net_operating_income` | float | Net operating income of the borrower | Required |
| `total_debt_service` | float | Total debt service obligations (principal, interest, lease payments, etc.) | Required |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `dscr` | float | The calculated Debt Service Coverage Ratio |
| `assessment` | str | Text description of the risk level |
| `rating` | str | Categorical rating ("Poor", "Fair", "Good", or "Excellent") |

## Risk Level Classification

The DSCR is categorized into risk levels:

| DSCR Range | Risk Level |
|------------|------------|
| < 1.0 | Poor |
| 1.0 - 1.25 | Fair |
| 1.25 - 1.5 | Good |
| > 1.5 | Excellent |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze DSCR for different borrowers:

```python
from pypulate.credit import debt_service_coverage_ratio
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Strong borrower with high income relative to debt
strong_borrower = debt_service_coverage_ratio(
    net_operating_income=800000,  # $800,000 net operating income
    total_debt_service=400000     # $400,000 total debt service
)

# Example 2: Average borrower with moderate coverage
average_borrower = debt_service_coverage_ratio(
    net_operating_income=450000,  # $450,000 net operating income
    total_debt_service=350000     # $350,000 total debt service
)

# Example 3: Weak borrower with insufficient coverage
weak_borrower = debt_service_coverage_ratio(
    net_operating_income=280000,  # $280,000 net operating income
    total_debt_service=300000     # $300,000 total debt service
)

# Print the results
print("Debt Service Coverage Ratio Analysis")
print("===================================")

print("\nExample 1: Strong Borrower")
print(f"DSCR: {strong_borrower['dscr']:.2f}")
print(f"Assessment: {strong_borrower['assessment']}")
print(f"Rating: {strong_borrower['rating']}")

print("\nExample 2: Average Borrower")
print(f"DSCR: {average_borrower['dscr']:.2f}")
print(f"Assessment: {average_borrower['assessment']}")
print(f"Rating: {average_borrower['rating']}")

print("\nExample 3: Weak Borrower")
print(f"DSCR: {weak_borrower['dscr']:.2f}")
print(f"Assessment: {weak_borrower['assessment']}")
print(f"Rating: {weak_borrower['rating']}")

# Visualize the results
borrowers = ['Weak', 'Average', 'Strong']
dscr_values = [
    weak_borrower['dscr'],
    average_borrower['dscr'],
    strong_borrower['dscr']
]

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(borrowers, dscr_values, color=['red', 'orange', 'green'])

# Add horizontal lines for the DSCR thresholds
plt.axhline(y=1.0, color='r', linestyle='--', label='Poor/Fair Threshold (1.0)')
plt.axhline(y=1.25, color='orange', linestyle='--', label='Fair/Good Threshold (1.25)')
plt.axhline(y=1.5, color='g', linestyle='--', label='Good/Excellent Threshold (1.5)')

# Add labels and title
plt.ylabel('Debt Service Coverage Ratio')
plt.title('DSCR Comparison')
plt.ylim(bottom=0)  # Start y-axis at 0

# Add the DSCR values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.show()

# Create a sensitivity analysis
income_values = np.linspace(200000, 1000000, 100)  # Range of income values
debt_service = 400000  # Fixed debt service

dscr_values = [income / debt_service for income in income_values]
ratings = []

for dscr in dscr_values:
    if dscr < 1.0:
        ratings.append("Poor")
    elif dscr < 1.25:
        ratings.append("Fair")
    elif dscr < 1.5:
        ratings.append("Good")
    else:
        ratings.append("Excellent")

# Create a plot showing how DSCR changes with income
plt.figure(figsize=(12, 6))

# Plot DSCR curve
plt.plot(income_values, dscr_values, 'b-', linewidth=2)

# Add colored regions for different ratings
poor_indices = [i for i, r in enumerate(ratings) if r == "Poor"]
fair_indices = [i for i, r in enumerate(ratings) if r == "Fair"]
good_indices = [i for i, r in enumerate(ratings) if r == "Good"]
excellent_indices = [i for i, r in enumerate(ratings) if r == "Excellent"]

if poor_indices:
    plt.fill_between(income_values[min(poor_indices):max(poor_indices)+1], 
                     0, dscr_values[min(poor_indices):max(poor_indices)+1], 
                     color='red', alpha=0.3, label='Poor')
if fair_indices:
    plt.fill_between(income_values[min(fair_indices):max(fair_indices)+1], 
                     0, dscr_values[min(fair_indices):max(fair_indices)+1], 
                     color='orange', alpha=0.3, label='Fair')
if good_indices:
    plt.fill_between(income_values[min(good_indices):max(good_indices)+1], 
                     0, dscr_values[min(good_indices):max(good_indices)+1], 
                     color='yellow', alpha=0.3, label='Good')
if excellent_indices:
    plt.fill_between(income_values[min(excellent_indices):max(excellent_indices)+1], 
                     0, dscr_values[min(excellent_indices):max(excellent_indices)+1], 
                     color='green', alpha=0.3, label='Excellent')

# Add horizontal lines for the DSCR thresholds
plt.axhline(y=1.0, color='r', linestyle='--')
plt.axhline(y=1.25, color='orange', linestyle='--')
plt.axhline(y=1.5, color='g', linestyle='--')

# Add labels and title
plt.xlabel('Net Operating Income ($)')
plt.ylabel('Debt Service Coverage Ratio')
plt.title('DSCR Sensitivity to Income (Fixed Debt Service: $400,000)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
```

## Example Output

```
Debt Service Coverage Ratio Analysis
===================================

Example 1: Strong Borrower
DSCR: 2.00
Assessment: Strong coverage, low risk
Rating: Excellent

Example 2: Average Borrower
DSCR: 1.29
Assessment: Sufficient coverage, acceptable risk
Rating: Good

Example 3: Weak Borrower
DSCR: 0.93
Assessment: Negative cash flow, high risk
Rating: Poor
```

## Visualizations

### DSCR Comparison

This visualization shows the DSCR values for three example borrowers, with horizontal lines indicating the threshold values that separate the rating categories.

### DSCR Sensitivity Analysis

This visualization demonstrates how the DSCR changes with varying levels of net operating income while keeping debt service constant, highlighting the income thresholds for different rating categories.

![DSCR Sensitivity](./charts/dscr.png)

## Practical Applications

The DSCR can be used for:

1. **Commercial Real Estate Lending**: Evaluating property cash flow relative to debt obligations
2. **Corporate Credit Analysis**: Assessing a company's ability to service its debt
3. **Project Finance**: Determining the financial viability of infrastructure projects
4. **Small Business Lending**: Evaluating loan applications from small businesses
5. **Risk-Based Pricing**: Setting interest rates based on the borrower's repayment capacity

## Industry Standards

Different lenders and industries may use slightly different DSCR thresholds:

1. **Commercial Real Estate**:
    - Typically requires DSCR ≥ 1.25
    - Premium properties may require DSCR ≥ 1.5
    - Riskier properties may accept DSCR ≥ 1.15

2. **Corporate Lending**:
    - Investment grade: DSCR ≥ 1.5
    - Non-investment grade: DSCR ≥ 1.2
    - Distressed: DSCR < 1.0

3. **Small Business Administration (SBA)**:
    - Generally requires DSCR ≥ 1.15
    - May consider global DSCR including owner's personal income

## Best Practices

1. **Historical Analysis**: Calculate DSCR using historical data to establish trends
2. **Stress Testing**: Test DSCR under adverse scenarios (e.g., reduced income, increased interest rates)
3. **Industry Comparison**: Compare DSCR to industry benchmarks
4. **Global DSCR**: Consider all sources of income and all debt obligations
5. **Forward-Looking**: Project future DSCR based on expected changes in income and debt
