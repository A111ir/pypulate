# Logistic Regression Score

The `logistic_regression_score` function implements a credit scoring model based on logistic regression, which is widely used in credit risk assessment. This function converts logistic regression outputs into a credit score on a standard scale (300-850), making it easier to interpret and use in credit decisions.

## Usage in Pypulate

```python
from pypulate.credit import logistic_regression_score

# Calculate credit score using logistic regression
result = logistic_regression_score(
    coefficients=[0.5, -0.3, 0.8, -0.4],  # Coefficients from logistic regression model
    features=[25000, 0.3, 5, 2],          # Feature values (e.g., income, DTI, years employed, inquiries)
    intercept=-2.5                        # Intercept term from logistic regression model
)

# Access the results
probability = result["probability_of_default"]
score = result["credit_score"]
risk_category = result["risk_category"]
log_odds = result["log_odds"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `coefficients` | array_like | Coefficients from the logistic regression model | Required |
| `features` | array_like | Feature values for the borrower being scored | Required |
| `intercept` | float | Intercept term from the logistic regression model | 0 |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `probability_of_default` | float | The calculated probability of default (between 0 and 1) |
| `credit_score` | int | The credit score on a 300-850 scale |
| `risk_category` | str | Categorical risk assessment ("Excellent", "Good", "Fair", "Poor", or "Very Poor") |
| `log_odds` | float | The log odds from the logistic regression calculation |

## Risk Level Classification

The credit score is categorized into risk levels:

| Credit Score Range | Risk Level |
|-------------------|------------|
| 750-850 | Excellent |
| 700-749 | Good |
| 650-699 | Fair |
| 600-649 | Poor |
| 300-599 | Very Poor |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze credit scores for different borrowers:

```python
from pypulate.credit import logistic_regression_score
import matplotlib.pyplot as plt
import numpy as np

# Define a simple logistic regression model
# Coefficients for: income (in $10k), DTI ratio, years employed, recent inquiries
coefficients = [
    -0.2,  # Income (negative coefficient: higher income -> lower default probability)
    2.5,   # DTI ratio (positive coefficient: higher DTI -> higher default probability)
    -0.3,  # Years employed (negative coefficient: longer employment -> lower default probability)
    0.4    # Recent inquiries (positive coefficient: more inquiries -> higher default probability)
]
intercept = -1.0  # Intercept term

# Example 1: Low-risk borrower
low_risk_borrower = logistic_regression_score(
    coefficients=coefficients,
    features=[8.0, 0.25, 10, 0],  # $80k income, 25% DTI, 10 years employed, 0 inquiries
    intercept=intercept
)

# Example 2: Medium-risk borrower
medium_risk_borrower = logistic_regression_score(
    coefficients=coefficients,
    features=[4.5, 0.42, 3, 2],  # $45k income, 42% DTI, 3 years employed, 2 inquiries
    intercept=intercept
)

# Example 3: High-risk borrower
high_risk_borrower = logistic_regression_score(
    coefficients=coefficients,
    features=[3.0, 0.45, 2, 4],  # $30k income, 45% DTI, 2 years employed, 4 inquiries
    intercept=intercept
)

# Print the results
print("Logistic Regression Credit Scoring Analysis")
print("==========================================")

print("\nExample 1: Low-Risk Borrower")
print(f"Credit Score: {low_risk_borrower['credit_score']}")
print(f"Probability of Default: {low_risk_borrower['probability_of_default']:.4f}")
print(f"Risk Category: {low_risk_borrower['risk_category']}")
print(f"Log Odds: {low_risk_borrower['log_odds']:.4f}")

print("\nExample 2: Medium-Risk Borrower")
print(f"Credit Score: {medium_risk_borrower['credit_score']}")
print(f"Probability of Default: {medium_risk_borrower['probability_of_default']:.4f}")
print(f"Risk Category: {medium_risk_borrower['risk_category']}")
print(f"Log Odds: {medium_risk_borrower['log_odds']:.4f}")

print("\nExample 3: High-Risk Borrower")
print(f"Credit Score: {high_risk_borrower['credit_score']}")
print(f"Probability of Default: {high_risk_borrower['probability_of_default']:.4f}")
print(f"Risk Category: {high_risk_borrower['risk_category']}")
print(f"Log Odds: {high_risk_borrower['log_odds']:.4f}")

# Visualize the results - Credit Score Comparison
risk_profiles = ['Low Risk', 'Medium Risk', 'High Risk']
scores = [
    low_risk_borrower['credit_score'],
    medium_risk_borrower['credit_score'],
    high_risk_borrower['credit_score']
]
probabilities = [
    low_risk_borrower['probability_of_default'],
    medium_risk_borrower['probability_of_default'],
    high_risk_borrower['probability_of_default']
]

# Create a bar chart for credit score comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(risk_profiles, scores, color=['green', 'orange', 'red'])

# Add horizontal lines for the score thresholds
plt.axhline(y=750, color='g', linestyle='--', label='Excellent (≥ 750)')
plt.axhline(y=700, color='b', linestyle='--', label='Good (≥ 700)')
plt.axhline(y=650, color='orange', linestyle='--', label='Fair (≥ 650)')
plt.axhline(y=600, color='r', linestyle='--', label='Poor (≥ 600)')

# Add the score values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.0f}', ha='center', va='bottom')

plt.ylabel('Credit Score')
plt.title('Credit Score Comparison')
plt.ylim(300, 850)  # Standard credit score range
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Create a figure showing the relationship between probability and score
plt.figure(figsize=(12, 6))
prob_range = np.linspace(0, 1, 100)
score_range = 850 - 550 * prob_range
score_range = np.clip(score_range, 300, 850)

plt.plot(prob_range, score_range, 'b-', linewidth=2)

# Add points for our examples
plt.scatter([low_risk_borrower['probability_of_default']], [low_risk_borrower['credit_score']], 
            color='green', s=100, label='Low Risk')
plt.scatter([medium_risk_borrower['probability_of_default']], [medium_risk_borrower['credit_score']], 
            color='orange', s=100, label='Medium Risk')
plt.scatter([high_risk_borrower['probability_of_default']], [high_risk_borrower['credit_score']], 
            color='red', s=100, label='High Risk')

# Add horizontal lines for score categories
plt.axhline(y=750, color='g', linestyle='--')
plt.axhline(y=700, color='b', linestyle='--')
plt.axhline(y=650, color='orange', linestyle='--')
plt.axhline(y=600, color='r', linestyle='--')

# Add text labels for score categories
plt.text(0.95, 800, 'Excellent', ha='right', va='center', color='green', fontweight='bold')
plt.text(0.95, 725, 'Good', ha='right', va='center', color='blue', fontweight='bold')
plt.text(0.95, 675, 'Fair', ha='right', va='center', color='orange', fontweight='bold')
plt.text(0.95, 625, 'Poor', ha='right', va='center', color='red', fontweight='bold')
plt.text(0.95, 450, 'Very Poor', ha='right', va='center', color='darkred', fontweight='bold')

plt.xlabel('Probability of Default')
plt.ylabel('Credit Score')
plt.title('Relationship Between Probability of Default and Credit Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Create a sensitivity analysis for a single feature
feature_index = 1  # DTI ratio (index 1 in our feature list)
feature_name = "Debt-to-Income Ratio"
feature_values = np.linspace(0.1, 0.6, 50)  # Range of DTI values from 10% to 60%
scores = []
probabilities = []

# Base features for a typical borrower
base_features = [5.0, 0.35, 5, 2]  # $50k income, 35% DTI, 5 years employed, 2 inquiries

for feature_value in feature_values:
    # Create a copy of base features and update the feature of interest
    test_features = base_features.copy()
    test_features[feature_index] = feature_value
    
    # Calculate score
    result = logistic_regression_score(
        coefficients=coefficients,
        features=test_features,
        intercept=intercept
    )
    scores.append(result['credit_score'])
    probabilities.append(result['probability_of_default'])

# Plot score sensitivity to feature
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(feature_values * 100, scores, 'b-', linewidth=2)
plt.xlabel(f'{feature_name} (%)')
plt.ylabel('Credit Score')
plt.title(f'Credit Score Sensitivity to {feature_name}')
plt.grid(True, linestyle='--', alpha=0.7)

# Add horizontal lines for score categories
plt.axhline(y=750, color='g', linestyle='--', label='Excellent (≥ 750)')
plt.axhline(y=700, color='b', linestyle='--', label='Good (≥ 700)')
plt.axhline(y=650, color='orange', linestyle='--', label='Fair (≥ 650)')
plt.axhline(y=600, color='r', linestyle='--', label='Poor (≥ 600)')
plt.legend(loc='lower left')

# Plot probability sensitivity to feature
plt.subplot(1, 2, 2)
plt.plot(feature_values * 100, probabilities, 'r-', linewidth=2)
plt.xlabel(f'{feature_name} (%)')
plt.ylabel('Probability of Default')
plt.title(f'Default Probability Sensitivity to {feature_name}')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

## Example Output

```
Logistic Regression Credit Scoring Analysis
==========================================
Example 1: Low-Risk Borrower
Credit Score: 847
Probability of Default: 0.0069
Risk Category: Excellent
Log Odds: -4.9750

Example 2: Medium-Risk Borrower
Credit Score: 697
Probability of Default: 0.2789
Risk Category: Fair
Log Odds: -0.9500

Example 3: High-Risk Borrower
Credit Score: 505
Probability of Default: 0.6283
Risk Category: Very Poor
Log Odds: 0.5250
```

## Visualizations

### Credit Score Comparison

The following visualization shows a comparison of credit scores across three different borrower profiles:

![Credit Score Comparison](./charts/logistic_comparison.png)

This chart displays the credit scores for low, medium, and high-risk borrowers, with horizontal lines indicating the threshold values that separate different credit quality categories.

### Probability to Score Relationship

The following visualization demonstrates the relationship between probability of default and credit score:

![Probability to Score Relationship](./charts/logistic_relationship.png)

This chart illustrates how the credit score decreases as the probability of default increases, with points showing where our example borrowers fall on the curve.

### Feature Sensitivity Analysis

The following visualization shows how changes in a single feature affect both the credit score and probability of default:

![Feature Sensitivity Analysis](./charts/logistic_sensitivity.png)

This sensitivity analysis demonstrates how increasing the debt-to-income ratio leads to lower credit scores and higher default probabilities.
## Practical Applications

Logistic regression scoring can be used for:

1. **Credit Underwriting**: Automating credit decisions based on objective criteria
2. **Risk-Based Pricing**: Setting interest rates based on creditworthiness
3. **Portfolio Segmentation**: Dividing borrowers into risk tiers for targeted strategies
4. **Pre-qualification**: Providing potential borrowers with preliminary credit assessments
5. **Account Management**: Monitoring existing customers for changes in credit quality

## Advantages and Limitations

### Advantages

1. **Interpretability**: Coefficients directly show the impact of each feature
2. **Probability Output**: Provides a meaningful probability of default
3. **Efficiency**: Computationally simple and fast to implement
4. **Flexibility**: Can incorporate various types of features
5. **Standard Scale**: Converts to a familiar credit score scale

### Limitations

1. **Linearity Assumption**: Assumes a linear relationship in the log odds
2. **Feature Independence**: Doesn't naturally capture interactions between features
3. **Data Quality Dependency**: Performance depends on the quality of training data
4. **Model Simplicity**: May not capture complex patterns as well as more advanced models
5. **Calibration Needs**: Requires proper calibration to produce accurate probabilities 