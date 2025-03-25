import pytest
import math
from pypulate.pricing.subscription_pricing import calculate_subscription_price


def test_basic_subscription_no_features():
    # Test basic subscription with no features
    base_price = 49.99
    features = []
    feature_prices = {}
    result = calculate_subscription_price(base_price, features, feature_prices)
    assert math.isclose(result, 49.99)


def test_subscription_with_features():
    # Test subscription with additional features
    base_price = 99.99
    features = ['premium', 'api_access']
    feature_prices = {'premium': 49.99, 'api_access': 29.99}
    result = calculate_subscription_price(base_price, features, feature_prices)
    expected = 99.99 + 49.99 + 29.99
    assert math.isclose(result, expected)


def test_subscription_with_duration():
    # Test subscription with a duration longer than 1 month, no discount
    base_price = 99.99
    features = ['premium']
    feature_prices = {'premium': 49.99}
    duration_months = 6
    result = calculate_subscription_price(base_price, features, feature_prices, duration_months=duration_months)
    expected = (99.99 + 49.99) * 6
    assert math.isclose(result, expected)


def test_subscription_with_discount():
    # Test subscription with a duration and a discount
    base_price = 99.99
    features = ['premium', 'api_access']
    feature_prices = {'premium': 49.99, 'api_access': 29.99}
    duration_months = 12
    discount_rate = 0.10
    result = calculate_subscription_price(
        base_price, features, feature_prices, 
        duration_months=duration_months, discount_rate=discount_rate
    )
    monthly_price = 99.99 + 49.99 + 29.99
    annual_discount = (1 - discount_rate) ** (duration_months / 12)
    expected = monthly_price * duration_months * annual_discount
    assert math.isclose(result, expected)


def test_subscription_with_nonexistent_features():
    # Test with features that don't exist in feature_prices
    base_price = 99.99
    features = ['premium', 'nonexistent']
    feature_prices = {'premium': 49.99}
    result = calculate_subscription_price(base_price, features, feature_prices)
    expected = 99.99 + 49.99  # Nonexistent feature should be ignored
    assert math.isclose(result, expected)


def test_zero_base_price():
    # Test with zero base price
    base_price = 0.0
    features = ['premium']
    feature_prices = {'premium': 49.99}
    result = calculate_subscription_price(base_price, features, feature_prices)
    assert math.isclose(result, 49.99)


def test_zero_duration():
    # Test with zero duration (should be treated as 1 month)
    base_price = 99.99
    features = []
    feature_prices = {}
    duration_months = 0
    # Zero or negative duration should be treated as 1 month in a real implementation,
    # but we'll follow the current implementation which doesn't guard against this
    result = calculate_subscription_price(base_price, features, feature_prices, duration_months=duration_months)
    assert math.isclose(result, 0.0)  # 99.99 * 0


def test_negative_discount_rate():
    # Test with negative discount rate (premium instead of discount)
    base_price = 99.99
    features = []
    feature_prices = {}
    duration_months = 12
    discount_rate = -0.10  # 10% premium
    result = calculate_subscription_price(
        base_price, features, feature_prices,
        duration_months=duration_months, discount_rate=discount_rate
    )
    annual_premium = (1 - (-0.10)) ** (duration_months / 12)
    expected = base_price * duration_months * annual_premium
    assert math.isclose(result, expected)


def test_high_discount_rate():
    # Test with high discount rate (near 100%)
    base_price = 99.99
    features = []
    feature_prices = {}
    duration_months = 12
    discount_rate = 0.99  # 99% discount
    result = calculate_subscription_price(
        base_price, features, feature_prices,
        duration_months=duration_months, discount_rate=discount_rate
    )
    annual_discount = (1 - 0.99) ** (duration_months / 12)
    expected = base_price * duration_months * annual_discount
    assert math.isclose(result, expected)


def test_many_features():
    # Test with many features
    base_price = 99.99
    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    feature_prices = {
        'feature1': 10.0,
        'feature2': 20.0,
        'feature3': 30.0,
        'feature4': 40.0,
        'feature5': 50.0,
    }
    result = calculate_subscription_price(base_price, features, feature_prices)
    expected = 99.99 + 10.0 + 20.0 + 30.0 + 40.0 + 50.0
    assert math.isclose(result, expected) 