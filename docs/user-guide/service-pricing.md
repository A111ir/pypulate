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

1. **Tiered Pricing**:
   - Use cumulative pricing for fair billing across tiers
   - Ensure tier ranges are continuous without gaps
   - Use "+" suffix for unlimited upper tiers

2. **Subscription Pricing**:
   - Set reasonable discount rates for longer subscriptions
   - Keep feature prices proportional to their value
   - Consider minimum subscription durations

3. **Usage Pricing**:
   - Set appropriate minimum charges to cover fixed costs
   - Use maximum charges to make costs predictable
   - Choose meaningful usage metrics

4. **Dynamic Pricing**:
   - Keep market factors between 0.5 and 2.0
   - Set reasonable price floors and ceilings
   - Update factors regularly based on market conditions

5. **Custom Rules**:
   - Document rule logic clearly
   - Validate inputs in custom calculation functions
   - Consider rule interactions and precedence

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