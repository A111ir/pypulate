import pytest
from pypulate.pricing.loyalty_based_pricing import calculate_loyalty_price


def test_basic_loyalty_pricing():
    # Test basic case with single loyalty tier
    base_price = 100.0
    customer_tenure = 12  # 12 months
    loyalty_tiers = {6: 0.10}  # 10% discount for 6+ months
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 90.0  # 100 - (100 * 0.10)
    assert result['loyalty_tier'] == 6
    assert result['loyalty_discount'] == 10.0
    assert result['additional_benefits'] == {}


def test_multiple_loyalty_tiers():
    # Test with multiple loyalty tiers, customer qualifies for higher tier
    base_price = 200.0
    customer_tenure = 24  # 24 months
    loyalty_tiers = {6: 0.05, 12: 0.10, 24: 0.15}  # increasing discounts
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 170.0  # 200 - (200 * 0.15)
    assert result['loyalty_tier'] == 24
    assert result['loyalty_discount'] == 30.0
    assert result['additional_benefits'] == {}


def test_tenure_between_tiers():
    # Test with tenure between tiers (should use the lower tier)
    base_price = 150.0
    customer_tenure = 18  # 18 months (between 12 and 24)
    loyalty_tiers = {6: 0.05, 12: 0.10, 24: 0.15}
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 135.0  # 150 - (150 * 0.10)
    assert result['loyalty_tier'] == 12
    assert result['loyalty_discount'] == 15.0
    assert result['additional_benefits'] == {}


def test_tenure_below_any_tier():
    # Test with tenure below any tier (should get no discount)
    base_price = 100.0
    customer_tenure = 3  # 3 months
    loyalty_tiers = {6: 0.05, 12: 0.10}
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 100.0  # No discount
    assert result['loyalty_tier'] == 0
    assert result['loyalty_discount'] == 0.0
    assert result['additional_benefits'] == {}


def test_empty_loyalty_tiers():
    # Test with empty loyalty tiers dictionary
    base_price = 100.0
    customer_tenure = 12
    loyalty_tiers = {}
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 100.0  # No discount
    assert result['loyalty_tier'] == 0
    assert result['loyalty_discount'] == 0.0
    assert result['additional_benefits'] == {}


def test_with_additional_benefits():
    # Test with additional benefits
    base_price = 100.0
    customer_tenure = 12
    loyalty_tiers = {6: 0.10}
    additional_benefits = {'free_shipping': 5.0, 'bonus_points': 10.0}
    result = calculate_loyalty_price(
        base_price, customer_tenure, loyalty_tiers, additional_benefits
    )
    
    assert result['loyalty_price'] == 90.0  # 100 - (100 * 0.10)
    assert result['loyalty_tier'] == 6
    assert result['loyalty_discount'] == 10.0
    assert result['additional_benefits'] == additional_benefits
    assert result['additional_benefits']['free_shipping'] == 5.0
    assert result['additional_benefits']['bonus_points'] == 10.0


def test_zero_base_price():
    # Test with zero base price
    base_price = 0.0
    customer_tenure = 12
    loyalty_tiers = {6: 0.10}
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 0.0
    assert result['loyalty_tier'] == 6
    assert result['loyalty_discount'] == 0.0
    assert result['additional_benefits'] == {}


def test_unsorted_loyalty_tiers():
    # Test with unsorted loyalty tiers
    base_price = 100.0
    customer_tenure = 24
    loyalty_tiers = {24: 0.15, 6: 0.05, 12: 0.10}  # Deliberately unsorted
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 85.0  # 100 - (100 * 0.15)
    assert result['loyalty_tier'] == 24
    assert result['loyalty_discount'] == 15.0
    assert result['additional_benefits'] == {}


def test_high_discount_rate():
    # Test with a high discount rate (nearly 100%)
    base_price = 100.0
    customer_tenure = 60
    loyalty_tiers = {60: 0.99}  # 99% discount
    result = calculate_loyalty_price(base_price, customer_tenure, loyalty_tiers)
    
    assert result['loyalty_price'] == 1.0  # 100 - (100 * 0.99)
    assert result['loyalty_tier'] == 60
    assert result['loyalty_discount'] == 99.0
    assert result['additional_benefits'] == {} 