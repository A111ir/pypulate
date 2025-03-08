# Credit Rating Transition Matrix

The `transition_matrix` function calculates a credit rating transition matrix, which shows the probability of credit ratings migrating from one level to another over a specified time period. This is a fundamental tool in credit risk management for understanding rating stability and modeling future rating changes.

## Usage in Pypulate

```python
from pypulate.credit import transition_matrix

# Calculate transition matrix from historical rating data
result = transition_matrix(
    ratings_t0=['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'],  # Ratings at time 0
    ratings_t1=['AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']     # Ratings at time 1
)

# Access the results
prob_matrix = result["probability_matrix"]
trans_matrix = result["transition_matrix"]
ratings = result["ratings"]
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ratings_t0` | array_like | Array of credit ratings at the beginning of the period | Required |
| `ratings_t1` | array_like | Array of credit ratings at the end of the period | Required |

## Return Value

The function returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `transition_matrix` | list of lists | Count of transitions from each rating to each other rating |
| `probability_matrix` | list of lists | Probability of transitioning from each rating to each other rating |
| `ratings` | list | Unique ratings found in the input data |

## Risk Level Classification

Credit ratings are typically categorized into risk levels:

| Rating Category | Risk Level |
|-----------------|------------|
| AAA, AA | Investment Grade - Very Low Risk |
| A, BBB | Investment Grade - Low to Moderate Risk |
| BB, B | Non-Investment Grade - Moderate to High Risk |
| CCC, CC, C | Non-Investment Grade - Very High Risk |
| D | Default |

## Comprehensive Example

Here's a complete example demonstrating how to calculate and analyze credit rating transitions:

```python
from pypulate.credit import transition_matrix
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Sample historical rating data
# Let's create a dataset with 100 companies and their ratings over two periods
np.random.seed(42)  # For reproducibility

# Define possible ratings
ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

# Generate initial ratings with a distribution skewed toward investment grade
weights = [0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05, 0.00]  # No defaults at t0
initial_ratings = np.random.choice(ratings, size=100, p=weights)

# Create a simple model for rating transitions
# Higher ratings are more stable, lower ratings have higher probability of downgrade
def generate_next_rating(current_rating):
    current_idx = ratings.index(current_rating)
    
    if current_rating == 'D':
        return 'D'  # Default is an absorbing state
    
    # Probability of staying at the same rating
    stay_prob = 0.7 - 0.05 * current_idx  # Higher ratings are more stable
    
    # Probability of upgrading (less likely for higher ratings)
    if current_idx == 0:
        upgrade_prob = 0  # AAA can't be upgraded
    else:
        upgrade_prob = 0.05 * (8 - current_idx) / 7  # More room for upgrade at lower ratings
    
    # Probability of downgrading (more likely for lower ratings)
    downgrade_prob = 1 - stay_prob - upgrade_prob
    
    # Determine direction of movement
    r = np.random.random()
    if r < stay_prob:
        return current_rating
    elif r < stay_prob + upgrade_prob:
        return ratings[current_idx - 1]  # Upgrade
    else:
        # For downgrades, allow for multi-notch downgrades for lower ratings
        if current_idx >= 5:  # B or lower
            # Possible to skip directly to default
            possible_downgrades = ratings[current_idx+1:]
            return np.random.choice(possible_downgrades)
        else:
            return ratings[current_idx + 1]  # Single notch downgrade

# Generate ratings at time 1
final_ratings = [generate_next_rating(rating) for rating in initial_ratings]

# Calculate the transition matrix
result = transition_matrix(initial_ratings, final_ratings)

# Print the transition probability matrix
print("Credit Rating Transition Matrix (Probabilities)")
print("==============================================")
prob_matrix = result['probability_matrix']
unique_ratings = result['ratings']

# Print the matrix with proper formatting
print(f"{'':5}", end="")
for r in unique_ratings:
    print(f"{r:6}", end="")
print()

for i, row in enumerate(prob_matrix):
    print(f"{unique_ratings[i]:5}", end="")
    for val in row:
        print(f"{val:.2f}  ", end="")
    print()

# Calculate some statistics
downgrades = sum(1 for i, j in zip(initial_ratings, final_ratings) 
                if ratings.index(j) > ratings.index(i))
upgrades = sum(1 for i, j in zip(initial_ratings, final_ratings) 
              if ratings.index(j) < ratings.index(i))
same = sum(1 for i, j in zip(initial_ratings, final_ratings) if i == j)
defaults = sum(1 for rating in final_ratings if rating == 'D')

print("\nTransition Statistics:")
print(f"Total Entities: {len(initial_ratings)}")
print(f"Upgrades: {upgrades} ({upgrades/len(initial_ratings):.1%})")
print(f"Downgrades: {downgrades} ({downgrades/len(initial_ratings):.1%})")
print(f"Unchanged: {same} ({same/len(initial_ratings):.1%})")
print(f"Defaults: {defaults} ({defaults/len(initial_ratings):.1%})")

# Visualize the transition matrix as a heatmap using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(prob_matrix, cmap='YlGnBu', aspect='equal')
plt.colorbar(label='Transition Probability')

# Add text annotations to the heatmap
for i in range(len(unique_ratings)):
    for j in range(len(unique_ratings)):
        plt.text(j, i, f"{prob_matrix[i][j]:.2f}", 
                 ha="center", va="center", 
                 color="black" if prob_matrix[i][j] < 0.7 else "white")

# Set ticks and labels
plt.xticks(range(len(unique_ratings)), unique_ratings)
plt.yticks(range(len(unique_ratings)), unique_ratings)
plt.xlabel('Rating at Time 1')
plt.ylabel('Rating at Time 0')
plt.title('Credit Rating Transition Matrix')
plt.tight_layout()
plt.show()

# Visualize rating distribution before and after
plt.figure(figsize=(12, 6))

# Count ratings at each time period
t0_counts = {r: 0 for r in ratings}
t1_counts = {r: 0 for r in ratings}

# Update with actual counts
for r in initial_ratings:
    t0_counts[r] += 1
for r in final_ratings:
    t1_counts[r] += 1

# Create bar chart
x = np.arange(len(ratings))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, [t0_counts[r] for r in ratings], width, label='Time 0')
ax.bar(x + width/2, [t1_counts[r] for r in ratings], width, label='Time 1')

# Add labels and legend
ax.set_xlabel('Credit Rating')
ax.set_ylabel('Number of Entities')
ax.set_title('Rating Distribution: Before vs After')
ax.set_xticks(x)
ax.set_xticklabels(ratings)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Calculate and visualize the default rates by initial rating
default_rates = []
for rating in ratings[:-1]:  # Exclude 'D' as initial rating
    entities_with_rating = sum(1 for r in initial_ratings if r == rating)
    if entities_with_rating > 0:
        defaults_from_rating = sum(1 for i, j in zip(initial_ratings, final_ratings) 
                                  if i == rating and j == 'D')
        default_rate = defaults_from_rating / entities_with_rating
    else:
        default_rate = 0
    default_rates.append(default_rate)

plt.figure(figsize=(10, 6))
plt.bar(ratings[:-1], [rate * 100 for rate in default_rates], color='darkred')
plt.title('Default Rate by Initial Credit Rating')
plt.xlabel('Initial Credit Rating')
plt.ylabel('Default Rate (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

## Example Output

```
Credit Rating Transition Matrix (Probabilities)
==============================================
     A     AA    AAA   B     BB    BBB   CCC   D     
A    0.52  0.00  0.00  0.00  0.00  0.48  0.00  0.00  
AA   0.23  0.68  0.09  0.00  0.00  0.00  0.00  0.00  
AAA  0.00  0.33  0.67  0.00  0.00  0.00  0.00  0.00  
B    0.00  0.00  0.00  0.50  0.00  0.00  0.00  0.50  
BB   0.00  0.00  0.00  0.53  0.35  0.12  0.00  0.00  
BBB  0.00  0.00  0.00  0.00  0.33  0.67  0.00  0.00  
CCC  0.00  0.00  0.00  0.00  0.00  0.00  0.40  0.60  
D    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  

Transition Statistics:
Total Entities: 100
Upgrades: 4 (4.0%)
Downgrades: 40 (40.0%)
Unchanged: 56 (56.0%)
Defaults: 7 (7.0%)
```

## Visualizations

### Transition Matrix Heatmap

This visualization shows the probability of transitioning from one rating to another as a color-coded heatmap, with darker colors indicating higher probabilities.

![Transition Matrix Heatmap](./charts/transition_heatmap.png)


### Rating Distribution

This bar chart compares the distribution of ratings at the beginning and end of the period, showing how the overall credit quality of the portfolio has changed.

![Transition Matrix Heatmap](./charts/transition_distribution.png)


### Default Rates by Initial Rating

This visualization shows the relationship between initial credit rating and default rate, illustrating the higher default probabilities associated with lower credit ratings.

## Practical Applications

Credit rating transition matrices can be used for:

1. **Credit Portfolio Management**: Modeling the evolution of portfolio credit quality over time
2. **Economic Capital Calculation**: Estimating potential credit losses under various scenarios
3. **Pricing Credit-Sensitive Instruments**: Determining appropriate spreads for bonds and loans
4. **Stress Testing**: Analyzing the impact of economic downturns on credit quality
5. **Regulatory Compliance**: Meeting requirements for internal ratings-based approaches under Basel frameworks

## Methodological Considerations

When calculating transition matrices, several methodological issues should be considered:

1. **Time Horizon**: Transition probabilities depend on the length of the observation period (e.g., 1-year vs. 5-year)
2. **Rating Withdrawals**: How to handle entities whose ratings are withdrawn during the period
3. **Point-in-Time vs. Through-the-Cycle**: Whether to use ratings that reflect current conditions or long-term averages
4. **Cohort vs. Hazard Rate Method**: Different approaches to calculating transition probabilities
5. **Economic Conditions**: Transition probabilities vary across different phases of the economic cycle

## Limitations

Credit rating transition matrices have several limitations:

1. **Rating Stability Bias**: Rating agencies may be slow to change ratings, leading to underestimation of transition probabilities
2. **Limited History**: For newer rating categories or markets, historical data may be insufficient
3. **Non-Markovian Behavior**: Future rating changes may depend on rating history, not just current rating
4. **Heterogeneity Within Ratings**: Entities with the same rating may have different default probabilities
5. **Time Variation**: Transition probabilities change over time with economic conditions 