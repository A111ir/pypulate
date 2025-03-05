# Service Pricing

The `ServicePricing` class provides a unified interface for calculating various types of service pricing models. It supports tiered pricing, subscription-based pricing, usage-based pricing, dynamic pricing adjustments, volume discounts, and custom pricing rules.

## Quick Start

```python
from pypulate import ServicePricing

# Initialize pricing calculator
pricing = ServicePricing()

# Calculate tiered pricing
price = pricing.calculate_tiered_price(
    usage_units=1500,
    tiers={
        "0-1000": 0.10,
        "1001-2000": 0.08,
        "2001+": 0.05
    }
)
print(f"Total price: ${price:.2f}")  # Output: Total price: $140.02
```

## Features

### Tiered Pricing

Calculate prices based on usage tiers:

```python
tiers = {
    "0-1000": 0.10,    # $0.10 per unit for first 1000 units = $100
    "1001-2000": 0.08, # $0.08 per unit for next 500 units = $40
    "2001+": 0.05      # $0.05 per unit for 2001+ units
}

# Cumulative pricing (default)
price = pricing.calculate_tiered_price(1500, tiers)
# Result: $140.02

# Non-cumulative pricing
price = pricing.calculate_tiered_price(1500, tiers, cumulative=False)
# Result: $120.00
```

### Subscription Pricing

Calculate subscription prices with features and discounts:

```python
price = pricing.calculate_subscription_price(
    base_price=99.99,
    features=['premium', 'api_access'],
    feature_prices={'premium': 49.99, 'api_access': 29.99},
    duration_months=12,
    discount_rate=0.10
)
```

### Usage-Based Pricing

Calculate prices based on multiple usage metrics:

```python
usage_metrics = {'api_calls': 1000, 'storage_gb': 50}
metric_rates = {'api_calls': 0.001, 'storage_gb': 0.10}
price = pricing.calculate_usage_price(
    usage_metrics,
    metric_rates,
    minimum_charge=10.0,
    maximum_charge=1000.0
)
```

### Volume Discounts

Apply volume-based discounts:

```python
discount_tiers = {
    100: 0.05,   # 5% discount for 100+ units
    500: 0.10,   # 10% discount for 500+ units
    1000: 0.15   # 15% discount for 1000+ units
}
price = pricing.calculate_volume_discount(
    base_price=10.0,
    volume=750,
    discount_tiers=discount_tiers
)
```

### Time-Based Pricing

Calculate prices based on time duration with different units and rounding options:

```python
price = pricing.calculate_time_based_price(
    base_price=25.0,      # $25 per hour
    duration=2.5,         # 2.5 hours
    time_unit='hour',     # pricing unit (minute, hour, day)
    minimum_duration=1.0, # minimum billable duration
    rounding_method='up'  # round up to nearest unit
)
# Result: $63.00 (25.0 * 2.5 = 62.5, rounded up to 63)

# Using minutes as the time unit
price = pricing.calculate_time_based_price(
    base_price=0.50,      # $0.50 per minute
    duration=45,          # 45 minutes
    time_unit='minute'
)
# Result: $23.00 (0.50 * 45 = 22.5, rounded up to 23)
```

### Freemium Pricing

Calculate prices for freemium models with base features (free up to limits) and premium features:

```python
price = pricing.calculate_freemium_price(
    base_features=['storage', 'api_calls', 'users'],
    premium_features=['advanced_analytics', 'priority_support'],
    feature_usage={
        'storage': 150,           # GB
        'api_calls': 12000,
        'users': 25,
        'advanced_analytics': 100,
        'priority_support': 1
    },
    free_limits={
        'storage': 100,           # 100 GB free
        'api_calls': 10000,       # 10,000 calls free
        'users': 20               # 20 users free
    },
    overage_rates={
        'storage': 0.1,           # $0.1 per GB over limit
        'api_calls': 0.001,       # $0.001 per call over limit
        'users': 2.0,             # $2 per user over limit
        'advanced_analytics': 0.05, # $0.05 per usage unit
        'priority_support': 50.0   # $50 flat fee (usage=1)
    }
)
```

### Bundle Pricing

Calculate prices for bundled items with combination-specific discounts:

```python
price = pricing.calculate_bundle_price(
    items=['laptop', 'mouse', 'keyboard', 'monitor'],
    item_prices={
        'laptop': 1200.0,
        'mouse': 25.0,
        'keyboard': 50.0,
        'monitor': 200.0
    },
    bundle_discounts={
        'laptop+mouse': 0.05,                # 5% off laptop+mouse
        'keyboard+mouse': 0.10,              # 10% off keyboard+mouse
        'laptop+keyboard+mouse': 0.15,       # 15% off laptop+keyboard+mouse
        'laptop+monitor+keyboard+mouse': 0.20 # 20% off complete setup
    },
    minimum_bundle_size=2  # minimum items for discount eligibility
)
# Result: $1180.00 (20% discount on the complete bundle)
```

### Peak Pricing

Apply different rates based on peak and off-peak hours:

```python
price = pricing.calculate_peak_pricing(
    base_price=50.0,      # base price per unit
    usage_time="14:30",   # time of usage (2:30 PM)
    peak_hours={
        "monday": ("09:00", "17:00"),
        "tuesday": ("09:00", "17:00"),
        "wednesday": ("09:00", "17:00"),
        "thursday": ("09:00", "17:00"),
        "friday": ("09:00", "17:00"),
        "saturday": ("10:00", "15:00"),
        "sunday": ("10:00", "15:00")
    },
    peak_multiplier=1.5,      # 50% premium during peak hours
    off_peak_multiplier=0.8   # 20% discount during off-peak hours
)
# Result: $75.00 during peak hours (1.5 * $50)
# Result: $40.00 during off-peak hours (0.8 * $50)
```

### Loyalty Pricing

Calculate prices with loyalty discounts based on customer tenure:

```python
result = pricing.calculate_loyalty_price(
    base_price=100.0,
    customer_tenure=24,    # months
    loyalty_tiers={
        12: 0.05,          # 5% discount after 1 year
        24: 0.10,          # 10% discount after 2 years
        36: 0.15           # 15% discount after 3 years
    },
    additional_benefits={
        'free_shipping': 10.0,
        'priority_support': 15.0
    }
)
# Result is a dictionary with details:
# {
#   'loyalty_price': 90.0,           # $100 - 10% discount
#   'loyalty_tier': 24,              # 2-year tier
#   'loyalty_discount': 10.0,        # $10 discount
#   'additional_benefits': {'free_shipping': 10.0, 'priority_support': 15.0}
# }

print(f"Loyalty Price: ${result['loyalty_price']}")
```

### Dynamic Pricing

Adjust prices based on market factors:

```python
price = pricing.apply_dynamic_pricing(
    base_price=100.0,
    demand_factor=1.2,      # High demand
    competition_factor=0.9,  # Strong competition
    seasonality_factor=1.1,  # Peak season
    min_price=80.0,
    max_price=150.0
)
```

### Custom Pricing Rules

Create and apply custom pricing rules:

```python
# Add a custom holiday pricing rule
pricing.add_custom_pricing_rule(
    'holiday',
    lambda price, multiplier: price * multiplier,
    description="Applies holiday season multiplier"
)

# Apply the custom rule
holiday_price = pricing.apply_custom_pricing_rule('holiday', 100.0, 1.2)
# Result: $120.00
```

## Price History Tracking

The `ServicePricing` class automatically tracks pricing calculations:

```python
# Save current pricing state to history
pricing.save_current_pricing()

# Get pricing history
history = pricing.get_pricing_history()
```

Each history entry contains:
- Timestamp of the calculation
- Pricing details for each calculation type (tiered, subscription, usage, etc.)

## Best Practices

### 1. Tiered Pricing
- 1.1. Use cumulative pricing for fair billing across tiers
- 1.2. Ensure tier ranges are continuous without gaps
- 1.3. Use "+" suffix for unlimited upper tiers

### 2. Subscription Pricing
- 2.1. Set reasonable discount rates for longer subscriptions
- 2.2. Keep feature prices proportional to their value
- 2.3. Consider minimum subscription durations

### 3. Usage Pricing
- 3.1. Set appropriate minimum charges to cover fixed costs
- 3.2. Use maximum charges to make costs predictable
- 3.3. Choose meaningful usage metrics

### 4. Time-Based Pricing
- 4.1. Choose appropriate time units for your service (minute, hour, day)
- 4.2. Set minimum durations to avoid micro-billing
- 4.3. Consider different rounding methods based on industry standards

### 5. Freemium Pricing
- 5.1. Clearly separate base (free) and premium features
- 5.2. Set reasonable free limits that provide value but encourage upgrades
- 5.3. Price premium features based on their value proposition

### 6. Bundle Pricing
- 6.1. Create meaningful bundles that complement each other
- 6.2. Increase discount rates for larger bundles
- 6.3. Set minimum bundle sizes to prevent abuse

### 7. Peak Pricing
- 7.1. Define peak hours based on actual usage patterns
- 7.2. Set reasonable multipliers that reflect demand without alienating customers
- 7.3. Consider different peak hours for different days of the week

### 8. Loyalty Pricing
- 8.1. Create meaningful tenure tiers that reward long-term customers
- 8.2. Include additional benefits beyond just discounts
- 8.3. Ensure discounts scale appropriately with tenure

### 9. Dynamic Pricing
- 9.1. Keep market factors between 0.5 and 2.0
- 9.2. Set reasonable price floors and ceilings
- 9.3. Update factors regularly based on market conditions

### 10. Custom Rules
- 10.1. Document rule logic clearly
- 10.2. Validate inputs in custom calculation functions
- 10.3. Consider rule interactions and precedence

## Error Handling

The class includes robust error handling:

- Invalid tier ranges raise ValueError
- Missing custom rules raise KeyError
- Invalid metric names raise KeyError
- Negative prices raise ValueError

## Performance Considerations

- Pricing calculations are optimized for speed
- History tracking has minimal overhead
- Custom rules are cached for repeated use
- Large tier structures are handled efficiently 