# KPI Guide

The `KPI` class provides a comprehensive suite of methods for calculating and tracking business metrics. It maintains state to provide health assessments and trend analysis.

## Basic Usage

```python
from pypulate import KPI

# Initialize KPI tracker
kpi = KPI()

# Calculate basic metrics
churn = kpi.churn_rate(customers_start=1000, customers_end=950, new_customers=50)
retention = kpi.retention_rate(customers_start=1000, customers_end=950, new_customers=50)

# Get health assessment
health = kpi.health
```

## Customer Metrics

### Churn and Retention
```python
# Calculate churn rate
churn = kpi.churn_rate(
    customers_start=1000,  # Starting customer count
    customers_end=950,     # Ending customer count
    new_customers=50       # New customers acquired
)

# Calculate retention rate
retention = kpi.retention_rate(
    customers_start=1000,
    customers_end=950,
    new_customers=50
)
```

### Customer Lifetime Value
```python
clv = kpi.customer_lifetime_value(
    avg_revenue_per_customer=100,  # Monthly revenue per customer
    gross_margin=70,              # Gross margin percentage
    churn_rate_value=5,          # Monthly churn rate
    discount_rate=10             # Annual discount rate
)
```

## Financial Metrics

### Revenue Metrics
```python
# Calculate MRR
mrr = kpi.monthly_recurring_revenue(
    paying_customers=1000,
    avg_revenue_per_customer=50
)

# Calculate ARR
arr = kpi.annual_recurring_revenue(
    paying_customers=1000,
    avg_revenue_per_customer=50
)
```

### Cost Metrics
```python
# Calculate CAC
cac = kpi.customer_acquisition_cost(
    marketing_costs=50000,
    sales_costs=30000,
    new_customers=100
)

# Calculate ROI
roi = kpi.roi(
    revenue=150000,
    costs=100000
)
```

## Engagement Metrics

### Net Promoter Score
```python
nps = kpi.net_promoter_score(
    promoters=70,        # Customers rating 9-10
    detractors=10,       # Customers rating 0-6
    total_respondents=100
)
```

### Customer Satisfaction
```python
# Calculate CSAT
csat = kpi.customer_satisfaction_score(
    satisfaction_ratings=[4, 5, 3, 5, 4],
    max_rating=5
)

# Calculate Customer Effort Score
ces = kpi.customer_effort_score(
    effort_ratings=[2, 3, 1, 2, 4],
    max_rating=7
)
```

## Health Assessment

The `health` property provides a comprehensive assessment of business health based on all tracked metrics:

```python
health = kpi.health

# Health assessment structure
{
    'overall_score': 85.5,
    'status': 'Good',
    'components': {
        'churn_rate': {
            'score': 90.0,
            'status': 'Excellent'
        },
        'retention_rate': {
            'score': 85.0,
            'status': 'Good'
        },
        # ... other metrics
    }
}
```

### Health Score Components

The health score is calculated based on weighted components:

- **Customer Health (30%)**
  - Churn Rate
  - Retention Rate
  - LTV/CAC Ratio

- **Financial Health (30%)**
  - Gross Margin
  - ROI
  - Revenue Growth

- **Engagement Health (40%)**
  - NPS
  - CSAT
  - Feature Adoption

Each component is scored from 0-100 and assigned a status:
- Excellent: ≥ 90
- Good: ≥ 75
- Fair: ≥ 60
- Poor: ≥ 45
- Critical: < 45

## State Management

The KPI class maintains state for all calculated metrics in the `_state` dictionary. This allows for:
- Trend analysis
- Health assessment
- Historical comparison
- Metric correlation

```python
# Access stored metrics
stored_churn = kpi._state['churn_rate']
stored_retention = kpi._state['retention_rate']
```

## Best Practices

1. **Initialize Early**: Create the KPI instance at the start of your analysis
2. **Regular Updates**: Update metrics consistently for accurate trending
3. **Monitor Health**: Regularly check the health assessment
4. **Store History**: Consider saving state for long-term analysis
5. **Validate Inputs**: Ensure input data quality for accurate metrics
``` 