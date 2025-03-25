"""
Tests for the create_scorecard function.
"""

import pytest
from pypulate.credit.create_scorecard import create_scorecard


def test_basic_scorecard_creation():
    """Test basic scorecard creation with default parameters."""
    features = {"income": 75000, "age": 35, "credit_history": 5}
    weights = {"income": 0.5, "age": 0.2, "credit_history": 15}
    
    result = create_scorecard(features, weights)
    
    assert "total_score" in result
    assert "points_breakdown" in result
    assert "risk_category" in result
    assert "thresholds" in result
    
    assert set(result["points_breakdown"].keys()) == set(weights.keys())


def test_excellent_category():
    """Test with values that should result in 'Excellent' risk category."""
    features = {"income": 150000, "age": 45, "credit_history": 10}
    weights = {"income": 0.5, "age": 0.3, "credit_history": 20}
    
    result = create_scorecard(features, weights)
    
    assert result["risk_category"] == "Excellent"
    assert result["total_score"] >= result["thresholds"]["Excellent"]


def test_good_category():
    """Test with values that should result in 'Good' risk category."""
    features = {"income": 50000, "age": 30, "credit_history": 4}
    weights = {"income": 0.2, "age": 0.1, "credit_history": 8}
    
    result = create_scorecard(features, weights)
    
    # Print score for debugging
    print(f"Good category score: {result['total_score']}, thresholds: {result['thresholds']}")
    
    assert result["risk_category"] == "Good"
    assert result["total_score"] >= result["thresholds"]["Good"]
    assert result["total_score"] < result["thresholds"]["Excellent"]


def test_fair_category():
    """Test with values that should result in 'Fair' risk category."""
    features = {"income": 35000, "age": 25, "credit_history": 2}
    weights = {"income": 0.15, "age": 0.1, "credit_history": 6}
    
    result = create_scorecard(features, weights)
    
    # Print score for debugging
    print(f"Fair category score: {result['total_score']}, thresholds: {result['thresholds']}")
    
    assert result["risk_category"] == "Fair"
    assert result["total_score"] >= result["thresholds"]["Fair"]
    assert result["total_score"] < result["thresholds"]["Good"]


def test_poor_category():
    """Test with values that should result in 'Poor' risk category."""
    features = {"income": 30000, "age": 25, "credit_history": 2}
    weights = {"income": 0.1, "age": 0.05, "credit_history": 5}
    
    result = create_scorecard(features, weights)
    
    assert result["risk_category"] == "Poor"
    assert result["total_score"] >= result["thresholds"]["Poor"]
    assert result["total_score"] < result["thresholds"]["Fair"]


def test_very_poor_category():
    """Test with values that should result in 'Very Poor' risk category."""
    features = {"income": -5000, "age": 18, "credit_history": 0}
    weights = {"income": 0.01, "age": 0.001, "credit_history": 0.1}
    
    result = create_scorecard(features, weights)
    
    # Print score for debugging
    print(f"Very Poor category score: {result['total_score']}, thresholds: {result['thresholds']}")
    
    assert result["risk_category"] == "Very Poor"
    assert result["total_score"] < result["thresholds"]["Poor"]


def test_custom_offsets():
    """Test with custom offset values."""
    features = {"income": 75000, "age": 35, "credit_history": 5}
    weights = {"income": 0.5, "age": 0.2, "credit_history": 15}
    offsets = {"income": 50000, "age": 25, "credit_history": 2}
    
    result = create_scorecard(features, weights, offsets)
    
    # Calculate expected points manually
    expected_income_points = (75000 - 50000) * 0.5 / 100.0
    expected_age_points = (35 - 25) * 0.2 / 100.0
    expected_credit_points = (5 - 2) * 15 / 100.0
    
    assert result["points_breakdown"]["income"] == pytest.approx(expected_income_points)
    assert result["points_breakdown"]["age"] == pytest.approx(expected_age_points)
    assert result["points_breakdown"]["credit_history"] == pytest.approx(expected_credit_points)


def test_custom_scaling_factor():
    """Test with custom scaling factor."""
    features = {"income": 75000, "age": 35}
    weights = {"income": 0.5, "age": 0.2}
    scaling_factor = 200.0  # Double the default
    
    # With default scaling
    result_default = create_scorecard(features, weights)
    
    # With custom scaling
    result_custom = create_scorecard(features, weights, scaling_factor=scaling_factor)
    
    # Points should be halved with double the scaling factor
    assert result_custom["points_breakdown"]["income"] == pytest.approx(result_default["points_breakdown"]["income"] / 2)
    assert result_custom["points_breakdown"]["age"] == pytest.approx(result_default["points_breakdown"]["age"] / 2)


def test_custom_base_score():
    """Test with custom base score."""
    features = {"income": 75000, "age": 35}
    weights = {"income": 0.5, "age": 0.2}
    base_score = 500  # Different from default 600
    
    result = create_scorecard(features, weights, base_score=base_score)
    
    # Base score should affect total and thresholds
    assert result["total_score"] == pytest.approx(base_score + result["points_breakdown"]["income"] + result["points_breakdown"]["age"])
    assert all(threshold <= default_threshold for threshold, default_threshold in zip(result["thresholds"].values(), [750, 700, 650, 600]))


def test_missing_feature():
    """Test with a feature missing from weights."""
    features = {"income": 75000, "age": 35, "missing_feature": 10}
    weights = {"income": 0.5, "age": 0.2}
    
    result = create_scorecard(features, weights)
    
    # Missing feature should not be in points breakdown
    assert "missing_feature" not in result["points_breakdown"]
    assert set(result["points_breakdown"].keys()) == set(weights.keys())


def test_missing_weight():
    """Test with a weight missing from features."""
    features = {"income": 75000, "age": 35}
    weights = {"income": 0.5, "age": 0.2, "credit_history": 15}
    
    result = create_scorecard(features, weights)
    
    # Missing feature should not be in points breakdown
    assert "credit_history" not in result["points_breakdown"]
    assert set(result["points_breakdown"].keys()) == set(features.keys())


def test_empty_features():
    """Test with empty features."""
    features = {}
    weights = {"income": 0.5, "age": 0.2}
    base_score = 600
    
    result = create_scorecard(features, weights)
    
    # Points breakdown should be empty, total score should be base score
    assert result["points_breakdown"] == {}
    assert result["total_score"] == base_score


def test_negative_features():
    """Test with negative feature values."""
    features = {"income": -10000, "age": 35}
    weights = {"income": 0.5, "age": 0.2}
    
    result = create_scorecard(features, weights)
    
    # Negative features should yield negative points
    assert result["points_breakdown"]["income"] < 0


def test_thresholds_adjustment():
    """Test that thresholds are properly adjusted with scaling factor."""
    features = {"income": 75000}
    weights = {"income": 0.5}
    scaling_factor_1 = 100.0
    scaling_factor_2 = 200.0
    
    result_1 = create_scorecard(features, weights, scaling_factor=scaling_factor_1)
    result_2 = create_scorecard(features, weights, scaling_factor=scaling_factor_2)
    
    # Print thresholds for debugging
    print(f"Thresholds with scaling_factor_1={scaling_factor_1}: {result_1['thresholds']}")
    print(f"Thresholds with scaling_factor_2={scaling_factor_2}: {result_2['thresholds']}")
    
    # The adjustment logic in the test is incorrect
    # The adjustment should account for the reference_scaling (100.0) as used in create_scorecard
    for category in ["Excellent", "Good", "Fair", "Poor"]:
        # The calculation in create_scorecard: base_score + (threshold - base_score) / adjustment_factor
        # where adjustment_factor = reference_scaling / scaling_factor
        
        # So for scaling_factor_1 = 100.0, adjustment_factor_1 = 100.0 / 100.0 = 1.0
        # For scaling_factor_2 = 200.0, adjustment_factor_2 = 100.0 / 200.0 = 0.5
        
        # For category "Excellent", if threshold = 750:
        # threshold_1 = 600 + (750 - 600) / 1.0 = 750
        # threshold_2 = 600 + (750 - 600) / 0.5 = 600 + 300 = 900
        
        base_score = 600
        reference_threshold = {"Excellent": 750, "Good": 700, "Fair": 650, "Poor": 600}[category]
        
        adjustment_factor_1 = 100.0 / scaling_factor_1
        adjustment_factor_2 = 100.0 / scaling_factor_2
        
        threshold_1 = base_score + (reference_threshold - base_score) / adjustment_factor_1
        threshold_2 = base_score + (reference_threshold - base_score) / adjustment_factor_2
        
        assert result_1["thresholds"][category] == pytest.approx(threshold_1)
        assert result_2["thresholds"][category] == pytest.approx(threshold_2) 