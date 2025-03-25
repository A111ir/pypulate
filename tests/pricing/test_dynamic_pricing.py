import pytest
import math
from pypulate.pricing.dynamic_pricing import apply_dynamic_pricing, PricingRule


class TestApplyDynamicPricing:
    def test_basic_calculation(self):
        # Test the basic multiplication calculation
        result = apply_dynamic_pricing(100.0, 1.2, 0.9, 1.1)
        expected = 100.0 * 1.2 * 0.9 * 1.1
        assert math.isclose(result, expected)
    
    def test_default_seasonality(self):
        # Test with default seasonality_factor (should be 1.0)
        result = apply_dynamic_pricing(100.0, 1.2, 0.9)
        expected = 100.0 * 1.2 * 0.9 * 1.0
        assert math.isclose(result, expected)
    
    def test_min_price_boundary(self):
        # Test minimum price boundary
        result = apply_dynamic_pricing(100.0, 0.5, 0.5, min_price=40.0)
        assert result == 40.0  # Should be bounded by min_price
    
    def test_max_price_boundary(self):
        # Test maximum price boundary
        result = apply_dynamic_pricing(100.0, 2.0, 2.0, max_price=300.0)
        assert result == 300.0  # Should be bounded by max_price
    
    def test_both_boundaries(self):
        # Test with both min and max price boundaries
        # Within boundaries
        result1 = apply_dynamic_pricing(100.0, 1.2, 0.9, min_price=90.0, max_price=150.0)
        expected1 = 100.0 * 1.2 * 0.9
        assert math.isclose(result1, expected1)
        
        # Below min boundary
        result2 = apply_dynamic_pricing(100.0, 0.5, 0.5, min_price=50.0, max_price=150.0)
        assert result2 == 50.0
        
        # Above max boundary
        result3 = apply_dynamic_pricing(100.0, 2.0, 2.0, min_price=50.0, max_price=300.0)
        assert result3 == 300.0
    
    def test_zero_base_price(self):
        # Test with zero base price
        result = apply_dynamic_pricing(0.0, 1.5, 1.5)
        assert result == 0.0
    
    def test_negative_factors(self):
        # Test with negative factors (not recommended but should work mathematically)
        result = apply_dynamic_pricing(100.0, -0.5, -0.5)
        expected = 100.0 * (-0.5) * (-0.5)
        assert math.isclose(result, expected)


class TestPricingRule:
    def test_add_and_apply_rule(self):
        # Test adding a rule and applying it
        rules = PricingRule()
        
        def test_rule(base, multiplier):
            return base * multiplier
        
        rules.add_rule("test", test_rule, "A test rule")
        result = rules.apply_rule("test", 100.0, 1.5)
        assert result == 150.0
    
    def test_add_and_apply_rule_with_lambda(self):
        # Test adding a rule with lambda and applying it
        rules = PricingRule()
        rules.add_rule("lambda_test", lambda x, y: x * y, "A lambda test rule")
        result = rules.apply_rule("lambda_test", 100.0, 1.2)
        assert result == 120.0
    
    def test_add_rule_with_kwargs(self):
        # Test adding a rule that uses keyword arguments
        rules = PricingRule()
        
        def kwarg_rule(base, *, factor=1.0, extra=0.0):
            return base * factor + extra
        
        rules.add_rule("kwarg_test", kwarg_rule)
        result = rules.apply_rule("kwarg_test", 100.0, factor=1.5, extra=10.0)
        assert result == 160.0
    
    def test_rule_not_found(self):
        # Test applying a non-existent rule
        rules = PricingRule()
        with pytest.raises(KeyError) as excinfo:
            rules.apply_rule("nonexistent", 100.0)
        assert "not found" in str(excinfo.value)
    
    def test_get_rule_description(self):
        # Test getting a rule description
        rules = PricingRule()
        description = "Test rule description"
        rules.add_rule("test", lambda x: x, description)
        assert rules.get_rule_description("test") == description
    
    def test_description_not_found(self):
        # Test getting description for a non-existent rule
        rules = PricingRule()
        with pytest.raises(KeyError) as excinfo:
            rules.get_rule_description("nonexistent")
        assert "not found" in str(excinfo.value)
    
    def test_list_rules(self):
        # Test listing all rules
        rules = PricingRule()
        rules.add_rule("rule1", lambda x: x, "First rule")
        rules.add_rule("rule2", lambda x: x*2, "Second rule")
        
        rule_list = rules.list_rules()
        assert len(rule_list) == 2
        assert rule_list["rule1"] == "First rule"
        assert rule_list["rule2"] == "Second rule"
    
    def test_list_rules_empty(self):
        # Test listing rules when none exist
        rules = PricingRule()
        rule_list = rules.list_rules()
        assert rule_list == {}
    
    def test_complex_pricing_function(self):
        # Test with a more complex pricing function
        rules = PricingRule()
        
        def seasonal_pricing(base_price, season, demand_multiplier=1.0):
            season_factors = {"summer": 1.2, "winter": 1.5, "fall": 0.9, "spring": 1.0}
            return base_price * season_factors.get(season, 1.0) * demand_multiplier
        
        rules.add_rule("seasonal", seasonal_pricing, "Seasonal pricing adjustment")
        
        summer_price = rules.apply_rule("seasonal", 100.0, "summer", demand_multiplier=1.1)
        winter_price = rules.apply_rule("seasonal", 100.0, "winter", demand_multiplier=0.8)
        
        assert math.isclose(summer_price, 100.0 * 1.2 * 1.1)
        assert math.isclose(winter_price, 100.0 * 1.5 * 0.8) 