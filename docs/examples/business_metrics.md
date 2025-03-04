---
title: Business Metrics
---

# Business Metrics Examples

This guide demonstrates various business metrics analysis using the `pypulate` package.

## Customer Metrics

### Churn and Retention Analysis

```python
from pypulate.kpi import churn_rate, retention_rate

# Monthly customer data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
customers_start = [100, 110, 125, 135, 150, 160]
customers_end = [110, 125, 135, 150, 160, 175]
new_customers = [20, 25, 20, 30, 25, 30]

# Calculate churn and retention rates
churn_rates = []
retention_rates = []

for i in range(len(months)):
    churn = churn_rate(customers_start[i], customers_end[i], new_customers[i])
    retention = retention_rate(customers_start[i], customers_end[i], new_customers[i])
    
    churn_rates.append(churn)
    retention_rates.append(retention)
    
    print(f"{months[i]}: Churn Rate = {churn:.2f}%, Retention Rate = {retention:.2f}%")

# Average churn and retention
avg_churn = sum(churn_rates) / len(churn_rates)
avg_retention = sum(retention_rates) / len(retention_rates)

print(f"Average Churn Rate: {avg_churn:.2f}%")
print(f"Average Retention Rate: {avg_retention:.2f}%")
```

### Customer Lifetime Value Analysis

```python
from pypulate.kpi import customer_lifetime_value, customer_acquisition_cost, ltv_cac_ratio

# Customer metrics
avg_revenue_per_customer = 100  # $100 per month
gross_margin = 70               # 70% gross margin
churn_rate_value = 5            # 5% monthly churn rate
discount_rate = 10              # 10% annual discount rate

# Marketing and sales costs
marketing_costs = 50000         # $50,000 marketing costs
sales_costs = 30000             # $30,000 sales costs
new_customers = 200             # 200 new customers

# Calculate CLV and CAC
ltv = customer_lifetime_value(avg_revenue_per_customer, gross_margin, churn_rate_value, discount_rate)
cac = customer_acquisition_cost(marketing_costs, sales_costs, new_customers)
ratio = ltv_cac_ratio(ltv, cac)

print(f"Customer Lifetime Value (CLV): ${ltv:.2f}")
print(f"Customer Acquisition Cost (CAC): ${cac:.2f}")
print(f"LTV:CAC Ratio: {ratio:.2f}")

# Evaluate business health
if ratio > 3:
    print("Excellent LTV:CAC ratio (>3)")
elif ratio > 1:
    print("Good LTV:CAC ratio (>1)")
else:
    print("Poor LTV:CAC ratio (<1)")
```

## Revenue Metrics

### MRR and ARR Analysis

```python
from pypulate.kpi import monthly_recurring_revenue, annual_recurring_revenue

# Monthly customer and revenue data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
paying_customers = [100, 110, 125, 135, 150, 160]
avg_revenue = [50, 52, 53, 55, 56, 58]  # Average revenue per customer

# Calculate MRR and ARR
mrr_values = []
arr_values = []

for i in range(len(months)):
    mrr = monthly_recurring_revenue(paying_customers[i], avg_revenue[i])
    arr = annual_recurring_revenue(paying_customers[i], avg_revenue[i])
    
    mrr_values.append(mrr)
    arr_values.append(arr)
    
    print(f"{months[i]}: MRR = ${mrr:.2f}, ARR = ${arr:.2f}")

# MRR growth
mrr_growth = [(mrr_values[i] - mrr_values[i-1]) / mrr_values[i-1] * 100 for i in range(1, len(mrr_values))]
avg_mrr_growth = sum(mrr_growth) / len(mrr_growth)

print(f"Average Monthly MRR Growth: {avg_mrr_growth:.2f}%")
```

### Revenue Churn and Expansion Analysis

```python
from pypulate.kpi import revenue_churn_rate, expansion_revenue_rate

# Monthly revenue data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
revenue_start = [5000, 5500, 6000, 6600, 7200, 7800]
revenue_end = [5500, 6000, 6600, 7200, 7800, 8500]
new_revenue = [800, 900, 1000, 1100, 1200, 1300]
upsell_revenue = [200, 250, 300, 350, 400, 450]
cross_sell_revenue = [100, 120, 150, 180, 200, 250]

# Calculate revenue churn and expansion rates
revenue_churn_rates = []
expansion_rates = []

for i in range(len(months)):
    rev_churn = revenue_churn_rate(revenue_start[i], revenue_end[i], new_revenue[i])
    expansion = expansion_revenue_rate(upsell_revenue[i], cross_sell_revenue[i], revenue_start[i])
    
    revenue_churn_rates.append(rev_churn)
    expansion_rates.append(expansion)
    
    print(f"{months[i]}: Revenue Churn = {rev_churn:.2f}%, Expansion = {expansion:.2f}%")

# Net revenue retention
net_retention = [(revenue_end[i] - new_revenue[i]) / revenue_start[i] * 100 for i in range(len(months))]
avg_net_retention = sum(net_retention) / len(net_retention)

print(f"Average Net Revenue Retention: {avg_net_retention:.2f}%")
```

## User Engagement Metrics

### Customer Satisfaction and Effort Analysis

```python
from pypulate.kpi import customer_satisfaction_score, customer_effort_score, net_promoter_score
import numpy as np

# Sample survey data
csat_ratings = np.array([4, 5, 3, 5, 4, 5, 4, 3, 5, 4])  # 1-5 scale
ces_ratings = np.array([2, 3, 1, 2, 4, 2, 3, 2, 1, 2])   # 1-7 scale (lower is better)

# NPS data
promoters = 70      # Customers who rated 9-10
detractors = 10     # Customers who rated 0-6
passives = 20       # Customers who rated 7-8
total_respondents = promoters + passives + detractors

# Calculate satisfaction metrics
csat = customer_satisfaction_score(csat_ratings, max_rating=5)
ces = customer_effort_score(ces_ratings, max_rating=7)
nps = net_promoter_score(promoters, detractors, total_respondents)

print(f"Customer Satisfaction Score (CSAT): {csat:.2f}%")
print(f"Customer Effort Score (CES): {ces:.2f}")
print(f"Net Promoter Score (NPS): {nps:.2f}")

# Evaluate customer satisfaction
if csat > 80:
    print("Excellent CSAT (>80%)")
elif csat > 70:
    print("Good CSAT (>70%)")
else:
    print("Poor CSAT (<70%)")
```

### User Activity Analysis

```python
from pypulate.kpi import daily_active_users_ratio, monthly_active_users_ratio, stickiness_ratio

# Monthly user activity data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
total_users = [2000, 2200, 2500, 2800, 3100, 3500]
monthly_active_users = [1500, 1700, 2000, 2300, 2600, 3000]
daily_active_users = [500, 600, 700, 800, 900, 1100]

# Calculate user activity metrics
dau_ratios = []
mau_ratios = []
stickiness_ratios = []

for i in range(len(months)):
    dau_ratio = daily_active_users_ratio(daily_active_users[i], total_users[i])
    mau_ratio = monthly_active_users_ratio(monthly_active_users[i], total_users[i])
    stickiness = stickiness_ratio(daily_active_users[i], monthly_active_users[i])
    
    dau_ratios.append(dau_ratio)
    mau_ratios.append(mau_ratio)
    stickiness_ratios.append(stickiness)
    
    print(f"{months[i]}: DAU Ratio = {dau_ratio:.2f}%, MAU Ratio = {mau_ratio:.2f}%, Stickiness = {stickiness:.2f}%")

# Average stickiness
avg_stickiness = sum(stickiness_ratios) / len(stickiness_ratios)
print(f"Average Stickiness Ratio: {avg_stickiness:.2f}%")
```

## Financial Health Metrics

### Burn Rate and Runway Analysis

```python
from pypulate.kpi import burn_rate, runway, gross_margin

# Financial data
starting_capital = 1000000  # $1,000,000 starting capital
ending_capital = 700000     # $700,000 ending capital
months = 6                  # 6 months period
current_capital = 700000    # $700,000 current capital

# Revenue and costs
revenue = 150000            # $150,000 revenue
cost_of_goods_sold = 45000  # $45,000 COGS

# Calculate financial health metrics
monthly_burn = burn_rate(starting_capital, ending_capital, months)
runway_months = runway(current_capital, monthly_burn)
gm = gross_margin(revenue, cost_of_goods_sold)

print(f"Monthly Burn Rate: ${monthly_burn:.2f}")
print(f"Runway: {runway_months:.2f} months")
print(f"Gross Margin: {gm:.2f}%")

# Evaluate financial health
if runway_months > 18:
    print("Healthy runway (>18 months)")
elif runway_months > 12:
    print("Adequate runway (>12 months)")
else:
    print("Concerning runway (<12 months)")
```

## Growth Metrics

### Conversion and Virality Analysis

```python
from pypulate.kpi import conversion_rate, virality_coefficient, feature_adoption_rate

# Growth data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
total_visitors = [10000, 12000, 15000, 18000, 22000, 25000]
conversions = [500, 650, 800, 1000, 1300, 1500]
new_users = [400, 500, 600, 800, 1000, 1200]
invites_sent = [2000, 2500, 3000, 3500, 4000, 5000]
total_users = [2000, 2400, 2900, 3500, 4300, 5300]
users_adopting_feature = [600, 800, 1000, 1400, 1800, 2300]

# Calculate growth metrics
conversion_rates = []
virality_coefficients = []
feature_adoption_rates = []

for i in range(len(months)):
    conv_rate = conversion_rate(conversions[i], total_visitors[i])
    virality = virality_coefficient(new_users[i], invites_sent[i], total_users[i])
    feature_adoption = feature_adoption_rate(users_adopting_feature[i], total_users[i])
    
    conversion_rates.append(conv_rate)
    virality_coefficients.append(virality)
    feature_adoption_rates.append(feature_adoption)
    
    print(f"{months[i]}: Conversion = {conv_rate:.2f}%, Virality = {virality:.2f}, Feature Adoption = {feature_adoption:.2f}%")

# Evaluate virality
avg_virality = sum(virality_coefficients) / len(virality_coefficients)
if avg_virality > 1:
    print("Viral growth (coefficient > 1)")
else:
    print("Non-viral growth (coefficient < 1)")
``` 