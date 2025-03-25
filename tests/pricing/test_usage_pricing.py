import pytest
import math
from pypulate.pricing.usage_pricing import calculate_usage_price, calculate_volume_discount


class TestCalculateUsagePrice:
    def test_basic_usage_calculation(self):
        # Test basic usage calculation with multiple metrics
        usage_metrics = {'api_calls': 1000, 'storage_gb': 50}
        metric_rates = {'api_calls': 0.001, 'storage_gb': 0.10}
        result = calculate_usage_price(usage_metrics, metric_rates)
        expected = (1000 * 0.001) + (50 * 0.10)  # 1 + 5 = 6
        assert math.isclose(result, expected)
    
    def test_with_minimum_charge(self):
        # Test with minimum charge higher than calculated price
        usage_metrics = {'api_calls': 100}
        metric_rates = {'api_calls': 0.001}
        # Calculated price would be 0.1, but minimum_charge is 1.0
        result = calculate_usage_price(usage_metrics, metric_rates, minimum_charge=1.0)
        assert result == 1.0
    
    def test_with_maximum_charge(self):
        # Test with maximum charge lower than calculated price
        usage_metrics = {'storage_gb': 1000}
        metric_rates = {'storage_gb': 0.10}
        # Calculated price would be 100, but maximum_charge is 50
        result = calculate_usage_price(usage_metrics, metric_rates, maximum_charge=50.0)
        assert result == 50.0
    
    def test_with_both_min_max_charges(self):
        # Test with both minimum and maximum charges
        usage_metrics = {'api_calls': 10000}
        metric_rates = {'api_calls': 0.001}
        # Calculated price is 10, which is between min (5) and max (15)
        result = calculate_usage_price(
            usage_metrics, metric_rates, minimum_charge=5.0, maximum_charge=15.0
        )
        assert result == 10.0
        
        # Test when calculated price is below minimum
        usage_metrics = {'api_calls': 1000}
        # Calculated price is 1, which is below min (5)
        result = calculate_usage_price(
            usage_metrics, metric_rates, minimum_charge=5.0, maximum_charge=15.0
        )
        assert result == 5.0
        
        # Test when calculated price is above maximum
        usage_metrics = {'api_calls': 20000}
        # Calculated price is 20, which is above max (15)
        result = calculate_usage_price(
            usage_metrics, metric_rates, minimum_charge=5.0, maximum_charge=15.0
        )
        assert result == 15.0
    
    def test_missing_metric_rates(self):
        # Test when a metric is missing from rates
        usage_metrics = {'api_calls': 1000, 'storage_gb': 50, 'bandwidth_gb': 200}
        metric_rates = {'api_calls': 0.001, 'storage_gb': 0.10}  # bandwidth_gb missing
        result = calculate_usage_price(usage_metrics, metric_rates)
        expected = (1000 * 0.001) + (50 * 0.10) + (200 * 0)  # bandwidth charged at 0
        assert math.isclose(result, expected)
    
    def test_empty_metrics(self):
        # Test with empty metrics
        usage_metrics = {}
        metric_rates = {'api_calls': 0.001}
        result = calculate_usage_price(usage_metrics, metric_rates)
        assert result == 0.0
    
    def test_zero_rates(self):
        # Test with zero rates
        usage_metrics = {'api_calls': 1000}
        metric_rates = {'api_calls': 0.0}
        result = calculate_usage_price(usage_metrics, metric_rates)
        assert result == 0.0


class TestCalculateVolumeDiscount:
    def test_basic_volume_discount(self):
        # Test basic volume discount
        base_price = 10.0
        volume = 750
        discount_tiers = {100: 0.05, 500: 0.10, 1000: 0.15}
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        # Volume is 750, which falls in the 500-tier with 10% discount
        expected = 750 * 10.0 * (1 - 0.10)  # 6750.0
        assert math.isclose(result, expected)
    
    def test_below_any_tier(self):
        # Test when volume is below any tier
        base_price = 10.0
        volume = 50
        discount_tiers = {100: 0.05, 500: 0.10, 1000: 0.15}
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        # Volume is 50, which is below any tier, so no discount
        expected = 50 * 10.0 * (1 - 0.0)  # 500.0
        assert math.isclose(result, expected)
    
    def test_at_tier_threshold(self):
        # Test when volume is exactly at a tier threshold
        base_price = 10.0
        volume = 500
        discount_tiers = {100: 0.05, 500: 0.10, 1000: 0.15}
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        # Volume is 500, which is at the 500-tier with 10% discount
        expected = 500 * 10.0 * (1 - 0.10)  # 4500.0
        assert math.isclose(result, expected)
    
    def test_highest_tier(self):
        # Test when volume qualifies for highest tier
        base_price = 10.0
        volume = 2000
        discount_tiers = {100: 0.05, 500: 0.10, 1000: 0.15}
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        # Volume is 2000, which is above the 1000-tier with 15% discount
        expected = 2000 * 10.0 * (1 - 0.15)  # 17000.0
        assert math.isclose(result, expected)
    
    def test_empty_discount_tiers(self):
        # Test with empty discount tiers
        base_price = 10.0
        volume = 1000
        discount_tiers = {}
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        # No discount tiers, so no discount
        expected = 1000 * 10.0  # 10000.0
        assert math.isclose(result, expected)
    
    def test_unsorted_discount_tiers(self):
        # Test with unsorted discount tiers
        base_price = 10.0
        volume = 750
        discount_tiers = {1000: 0.15, 100: 0.05, 500: 0.10}  # deliberately unsorted
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        # Volume is 750, should still apply 10% discount from 500-tier
        expected = 750 * 10.0 * (1 - 0.10)  # 6750.0
        assert math.isclose(result, expected)
    
    def test_zero_base_price(self):
        # Test with zero base price
        base_price = 0.0
        volume = 1000
        discount_tiers = {100: 0.05, 500: 0.10, 1000: 0.15}
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        assert result == 0.0
    
    def test_zero_volume(self):
        # Test with zero volume
        base_price = 10.0
        volume = 0
        discount_tiers = {100: 0.05, 500: 0.10, 1000: 0.15}
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        assert result == 0.0 