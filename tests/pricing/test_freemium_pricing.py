import pytest
from pypulate.pricing.freemium_pricing import calculate_freemium_price


def test_no_overage_within_limits():
    # Test when usage for a base feature is at or below the free limit, and premium features always charge
    base_features = ["feature1"]
    premium_features = ["feature2"]
    feature_usage = {"feature1": 100, "feature2": 50}
    free_limits = {"feature1": 100}
    overage_rates = {"feature1": 2.0, "feature2": 3.0}
    # For base: usage equals limit so no overage; for premium: cost = 50 * 3 = 150
    expected = 150.0
    result = calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    assert result == expected


def test_base_overage_only():
    # Test when a base feature exceeds its free limit
    base_features = ["feature1"]
    premium_features = []
    feature_usage = {"feature1": 120}  # usage is 120, free limit is 100
    free_limits = {"feature1": 100}
    overage_rates = {"feature1": 2.5}
    # overage = 20, cost = 20 * 2.5 = 50.0
    expected = 50.0
    result = calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    assert result == expected


def test_premium_features_only():
    # Test when only premium features are applied
    base_features = []
    premium_features = ["feature2"]
    feature_usage = {"feature2": 40}
    free_limits = {}  # not used for premium
    overage_rates = {"feature2": 4.0}
    # cost = 40 * 4.0 = 160.0
    expected = 160.0
    result = calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    assert result == expected


def test_combined_base_and_premium():
    # Test combining both base and premium features
    base_features = ["feature1", "feature2"]
    premium_features = ["feature3"]
    feature_usage = {"feature1": 150, "feature2": 80, "feature3": 30}
    free_limits = {"feature1": 100, "feature2": 100}
    overage_rates = {"feature1": 1.5, "feature2": 2.0, "feature3": 2.0}
    # For base: feature1: (150-100)*1.5 = 75, feature2: no overage
    # For premium: feature3: 30 * 2.0 = 60
    # Total = 75 + 60 = 135
    expected = 135.0
    result = calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    assert result == expected


def test_missing_feature_usage():
    # Test when a feature in the lists is not present in feature_usage
    base_features = ["feature1"]
    premium_features = ["feature2"]
    feature_usage = {"feature2": 10}  # feature1 is missing
    free_limits = {"feature1": 50}
    overage_rates = {"feature1": 3.0, "feature2": 2.0}
    # Base: feature1 ignored, Premium: 10 * 2.0 = 20
    expected = 20.0
    result = calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    assert result == expected


def test_empty_features():
    # Test with empty features lists
    base_features = []
    premium_features = []
    feature_usage = {}
    free_limits = {}
    overage_rates = {}
    expected = 0.0
    result = calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    assert result == expected


def test_feature_missing_in_overage_rates():
    # Test when a feature is missing from overage_rates; it should be ignored
    base_features = ["feature1"]
    premium_features = ["feature2"]
    feature_usage = {"feature1": 120, "feature2": 30}
    free_limits = {"feature1": 100}
    # overage_rates for feature1 is missing, so base cost will not be counted;
    # For premium, cost = 30 * rate if available
    overage_rates = {"feature2": 5.0}
    # Expected: base cost = 0, premium = 30 * 5 = 150
    expected = 150.0
    result = calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    assert result == expected 