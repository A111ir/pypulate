# Merton Model

The `merton_model` function implements the Merton structural model of default, a fundamental approach in credit risk modeling that treats a company's equity as a call option on its assets. This model, developed by Robert C. Merton in 1974, provides a framework for estimating the probability of default based on the company's capital structure and asset volatility.

## Usage in Pypulate

```python
from pypulate.credit import merton_model

# Calculate default probability using the Merton model
result = merton_model(
    asset_value=1000000,        # $1,000,000 market value of assets
    debt_face_value=600000,     # $600,000 face value of debt
    asset_volatility=0.25,      # 25% annualized asset volatility
    risk_free_rate=0.03,        # 3% risk-free rate
    time_to_maturity=1.0        # 1 year to debt maturity
)

# Access the results
pd = result["probability_of_default"]
dd = result["distance_to_default"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `asset_value` | float | Market value of the company's assets | Required |
| `debt_face_value` | float | Face value of the company's debt | Required |
| `asset_volatility` | float | Volatility of assets (annualized) | Required |
| `risk_free_rate` | float | Risk-free interest rate | Required |
| `time_to_maturity` | float | Time to debt maturity in years | Required |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `probability_of_default` | float | Probability of default within the time horizon |
| `distance_to_default` | float | Number of standard deviations to default threshold |
| `d1` | float | First parameter in the Black-Scholes-Merton formula |
| `d2` | float | Second parameter in the Black-Scholes-Merton formula |

## Risk Level Classification

The probability of default is categorized into risk levels:

| Probability of Default Range | Risk Level |
|------------------------------|------------|
| < 0.5% | Very Low |
| 0.5% - 2% | Low |
| 2% - 5% | Moderate |
| 5% - 15% | High |
| > 15% | Very High |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze default probabilities for companies with different financial profiles:

```python
from pypulate.credit import merton_model
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Financially strong company
strong_company = merton_model(
    asset_value=1000000,        # $1,000,000 market value of assets
    debt_face_value=400000,     # $400,000 face value of debt
    asset_volatility=0.20,      # 20% annualized asset volatility
    risk_free_rate=0.03,        # 3% risk-free rate
    time_to_maturity=1.0        # 1 year to debt maturity
)

# Example 2: Average company
average_company = merton_model(
    asset_value=1000000,        # $1,000,000 market value of assets
    debt_face_value=600000,     # $600,000 face value of debt
    asset_volatility=0.30,      # 30% annualized asset volatility
    risk_free_rate=0.03,        # 3% risk-free rate
    time_to_maturity=1.0        # 1 year to debt maturity
)

# Example 3: Financially distressed company
distressed_company = merton_model(
    asset_value=1000000,        # $1,000,000 market value of assets
    debt_face_value=800000,     # $800,000 face value of debt
    asset_volatility=0.40,      # 40% annualized asset volatility
    risk_free_rate=0.03,        # 3% risk-free rate
    time_to_maturity=1.0        # 1 year to debt maturity
)

# Print the results
print("Merton Model Analysis")
print("====================")

print("\nExample 1: Financially Strong Company")
print(f"Probability of Default: {strong_company['probability_of_default']:.4%}")
print(f"Distance to Default: {strong_company['distance_to_default']:.2f}")

print("\nExample 2: Average Company")
print(f"Probability of Default: {average_company['probability_of_default']:.4%}")
print(f"Distance to Default: {average_company['distance_to_default']:.2f}")

print("\nExample 3: Financially Distressed Company")
print(f"Probability of Default: {distressed_company['probability_of_default']:.4%}")
print(f"Distance to Default: {distressed_company['distance_to_default']:.2f}")

# Create a DataFrame for visualization
companies = ['Strong', 'Average', 'Distressed']
pd_values = [
    strong_company['probability_of_default'],
    average_company['probability_of_default'],
    distressed_company['probability_of_default']
]
dd_values = [
    strong_company['distance_to_default'],
    average_company['distance_to_default'],
    distressed_company['distance_to_default']
]

# Create a bar chart for probability of default
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
bars = plt.bar(companies, [pd * 100 for pd in pd_values], color=['green', 'orange', 'red'])

# Add the PD values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}%', ha='center', va='bottom')

plt.ylabel('Probability of Default (%)')
plt.title('Probability of Default by Company Type')
plt.ylim(0, max([pd * 100 for pd in pd_values]) * 1.2)  # Add some space above the highest bar

# Create a bar chart for distance to default
plt.subplot(1, 2, 2)
bars = plt.bar(companies, dd_values, color=['green', 'orange', 'red'])

# Add the DD values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}', ha='center', va='bottom')

plt.ylabel('Distance to Default')
plt.title('Distance to Default by Company Type')
plt.tight_layout()
plt.show()

# Sensitivity analysis: Effect of leverage (debt-to-asset ratio) on PD
leverage_ratios = np.linspace(0.1, 0.95, 50)  # Debt-to-asset ratios from 10% to 95%
pd_by_leverage = []
dd_by_leverage = []

for leverage in leverage_ratios:
    debt = leverage * 1000000  # Debt face value based on leverage ratio
    result = merton_model(
        asset_value=1000000,
        debt_face_value=debt,
        asset_volatility=0.30,
        risk_free_rate=0.03,
        time_to_maturity=1.0
    )
    pd_by_leverage.append(result['probability_of_default'])
    dd_by_leverage.append(result['distance_to_default'])

# Plot the effect of leverage on PD and DD
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(leverage_ratios, [pd * 100 for pd in pd_by_leverage], 'b-', linewidth=2)
plt.xlabel('Leverage Ratio (Debt/Assets)')
plt.ylabel('Probability of Default (%)')
plt.title('Effect of Leverage on Default Probability')
plt.grid(True, linestyle='--', alpha=0.7)

# Add risk level regions
plt.axhspan(0, 0.5, alpha=0.2, color='green', label='Very Low Risk')
plt.axhspan(0.5, 2, alpha=0.2, color='lightgreen', label='Low Risk')
plt.axhspan(2, 5, alpha=0.2, color='yellow', label='Moderate Risk')
plt.axhspan(5, 15, alpha=0.2, color='orange', label='High Risk')
plt.axhspan(15, 100, alpha=0.2, color='red', label='Very High Risk')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(leverage_ratios, dd_by_leverage, 'r-', linewidth=2)
plt.xlabel('Leverage Ratio (Debt/Assets)')
plt.ylabel('Distance to Default')
plt.title('Effect of Leverage on Distance to Default')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Sensitivity analysis: Effect of asset volatility on PD
volatilities = np.linspace(0.1, 0.6, 50)  # Asset volatilities from 10% to 60%
pd_by_volatility = []
dd_by_volatility = []

for vol in volatilities:
    result = merton_model(
        asset_value=1000000,
        debt_face_value=600000,  # 60% leverage
        asset_volatility=vol,
        risk_free_rate=0.03,
        time_to_maturity=1.0
    )
    pd_by_volatility.append(result['probability_of_default'])
    dd_by_volatility.append(result['distance_to_default'])

# Plot the effect of asset volatility on PD and DD
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(volatilities, [pd * 100 for pd in pd_by_volatility], 'b-', linewidth=2)
plt.xlabel('Asset Volatility')
plt.ylabel('Probability of Default (%)')
plt.title('Effect of Asset Volatility on Default Probability')
plt.grid(True, linestyle='--', alpha=0.7)

# Add risk level regions
plt.axhspan(0, 0.5, alpha=0.2, color='green', label='Very Low Risk')
plt.axhspan(0.5, 2, alpha=0.2, color='lightgreen', label='Low Risk')
plt.axhspan(2, 5, alpha=0.2, color='yellow', label='Moderate Risk')
plt.axhspan(5, 15, alpha=0.2, color='orange', label='High Risk')
plt.axhspan(15, 100, alpha=0.2, color='red', label='Very High Risk')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(volatilities, dd_by_volatility, 'r-', linewidth=2)
plt.xlabel('Asset Volatility')
plt.ylabel('Distance to Default')
plt.title('Effect of Asset Volatility on Distance to Default')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

## Example Output

```
Merton Model Analysis
====================

Merton Model Analysis
====================
Example 1: Financially Strong Company
Probability of Default: 0.0002%
Distance to Default: 4.63

Example 2: Average Company
Probability of Default: 4.9191%
Distance to Default: 1.65

Example 3: Financially Distressed Company
Probability of Default: 33.2559%
Distance to Default: 0.43
```

## Visualizations

### Default Probability and Distance to Default

These visualizations show the probability of default and distance to default for three example companies with different financial profiles.

![Merton Distance](./charts/merton.png)


### Sensitivity to Leverage

This analysis demonstrates how the probability of default and distance to default change with increasing leverage (debt-to-asset ratio), highlighting the non-linear relationship between leverage and default risk.

![Merton Leverage](./charts/merton_leverage.png)

### Sensitivity to Asset Volatility

This analysis shows how the probability of default and distance to default are affected by changes in asset volatility, illustrating the importance of asset stability in credit risk assessment.

## Theoretical Background

The Merton model is based on the following assumptions:

1. The company's capital structure consists of equity and a single zero-coupon debt issue
2. The company's asset value follows a geometric Brownian motion
3. Default occurs only at debt maturity if the asset value falls below the face value of debt
4. Markets are perfect (no transaction costs, taxes, or bankruptcy costs)
5. The risk-free rate is constant

Under these assumptions, the company's equity can be viewed as a European call option on the company's assets with a strike price equal to the face value of debt. The probability of default is then calculated as the probability that the asset value will be below the face value of debt at maturity.

## Practical Applications

The Merton model can be used for:

1. **Credit Risk Assessment**: Estimating default probabilities for corporate borrowers
2. **Bond Pricing**: Determining credit spreads for corporate bonds
3. **Portfolio Management**: Assessing the credit risk of investment portfolios
4. **Regulatory Capital**: Calculating capital requirements for credit risk
5. **Early Warning System**: Identifying companies with increasing default risk

## Limitations

While the Merton model provides a theoretically sound framework for credit risk assessment, it has several limitations:

1. **Simplified Capital Structure**: Assumes a single zero-coupon debt issue
2. **Default Timing**: Assumes default can only occur at debt maturity
3. **Asset Value Unobservability**: Requires estimation of unobservable asset value and volatility
4. **Constant Volatility**: Assumes asset volatility is constant over time
5. **Perfect Markets**: Ignores transaction costs, taxes, and bankruptcy costs

## Extensions

Several extensions to the basic Merton model have been developed to address its limitations:

1. **KMV Model**: Uses an iterative procedure to estimate asset value and volatility
2. **Black-Cox Model**: Allows for default before maturity if asset value falls below a threshold
3. **Longstaff-Schwartz Model**: Incorporates stochastic interest rates
4. **Leland Model**: Accounts for bankruptcy costs and tax benefits of debt
5. **CreditGrades Model**: Incorporates a stochastic default barrier 