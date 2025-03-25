"""
Tests for the logistic_regression_score function.
"""

import pytest
import numpy as np
from pypulate.credit.logistic_regression_score import logistic_regression_score


def test_basic_calculation():
    """Test basic logistic regression score calculation."""
    coefficients = [0.01, -0.005, 0.02]
    features = [700, 60, 10]
    intercept = -2.5
    
    result = logistic_regression_score(coefficients, features, intercept)
    
    # Calculate log odds: coefficients • features + intercept
    expected_log_odds = 0.01 * 700 + (-0.005) * 60 + 0.02 * 10 + (-2.5)
    expected_log_odds = 7 - 0.3 + 0.2 - 2.5
    expected_log_odds = 4.4
    
    # Calculate probability: 1 / (1 + e^(-log_odds))
    expected_prob = 1 / (1 + np.exp(-expected_log_odds))
    
    # Calculate score: 850 - 550 * prob, capped between 300 and 850
    expected_score = 850 - int(550 * expected_prob)
    expected_score = max(300, min(850, expected_score))
    
    assert result["log_odds"] == pytest.approx(expected_log_odds)
    assert result["probability_of_default"] == pytest.approx(expected_prob)
    assert result["credit_score"] == expected_score
    assert "risk_category" in result


def test_probability_ranges():
    """Test calculation for different probability ranges."""
    # Very low probability (< 0.1)
    low_result = logistic_regression_score([0.005], [100], -3.0)
    assert low_result["probability_of_default"] < 0.1
    assert low_result["credit_score"] > 750
    
    # Medium probability (~0.5)
    med_result = logistic_regression_score([0.01], [0], 0.0)
    assert med_result["probability_of_default"] == pytest.approx(0.5, abs=0.01)
    assert 550 <= med_result["credit_score"] <= 600
    
    # High probability (> 0.9)
    high_result = logistic_regression_score([0.005], [100], 3.0)
    assert high_result["probability_of_default"] > 0.9
    assert high_result["credit_score"] < 400


def test_risk_categories():
    """Test all risk categories based on score ranges."""
    # Excellent: score >= 750
    excellent_result = logistic_regression_score([0.01], [10], -4.0)  # Very low probability
    assert excellent_result["credit_score"] >= 750
    assert excellent_result["risk_category"] == "Excellent"
    
    # Good: 700 <= score < 750
    good_result = logistic_regression_score([0.01], [10], -2.0)  # Low probability
    if 700 <= good_result["credit_score"] < 750:
        assert good_result["risk_category"] == "Good"
    
    # Fair: 650 <= score < 700
    fair_result = logistic_regression_score([0.01], [10], -1.0)  # Medium-low probability
    if 650 <= fair_result["credit_score"] < 700:
        assert fair_result["risk_category"] == "Fair"
    
    # Poor: 600 <= score < 650
    poor_result = logistic_regression_score([0.01], [10], 0.0)  # Medium probability
    if 600 <= poor_result["credit_score"] < 650:
        assert poor_result["risk_category"] == "Poor"
    
    # Very Poor: score < 600
    very_poor_result = logistic_regression_score([0.01], [10], 1.0)  # High probability
    if very_poor_result["credit_score"] < 600:
        assert very_poor_result["risk_category"] == "Very Poor"


def test_score_boundaries():
    """Test score calculation at category boundaries."""
    # Test score exactly at 750
    score_750 = find_score_with_target(750)
    assert score_750["credit_score"] == 750
    assert score_750["risk_category"] == "Excellent"
    
    # Test score exactly at 700
    score_700 = find_score_with_target(700)
    assert score_700["credit_score"] == 700
    assert score_700["risk_category"] == "Good"
    
    # Test score exactly at 650
    score_650 = find_score_with_target(650)
    assert score_650["credit_score"] == 650
    assert score_650["risk_category"] == "Fair"
    
    # Test score exactly at 600
    score_600 = find_score_with_target(600)
    assert score_600["credit_score"] == 600
    assert score_600["risk_category"] == "Poor"


def find_score_with_target(target_score):
    """Helper function to find an input that produces a target score."""
    # Calculate the probability that would result in the target score
    # score = 850 - int(550 * probability)
    # probability = (850 - score) / 550
    probability = (850 - target_score) / 550
    
    # Calculate the log_odds that would result in this probability
    # probability = 1 / (1 + e^(-log_odds))
    # log_odds = -ln((1/probability) - 1)
    log_odds = -np.log((1/probability) - 1)
    
    # Use a simple coefficient and feature, and set intercept to get the desired log_odds
    coefficients = [1.0]
    features = [0.0]
    intercept = log_odds
    
    return logistic_regression_score(coefficients, features, intercept)


def test_score_limits():
    """Test that score is properly limited to the 300-850 range."""
    # Very high probability (near 1), should be at or near the minimum score of 300
    very_high_prob = logistic_regression_score([0.1], [100], 10.0)
    assert very_high_prob["probability_of_default"] > 0.999
    assert very_high_prob["credit_score"] >= 300
    assert very_high_prob["credit_score"] <= 305  # Allow small rounding differences
    
    # Very low probability (near 0), should be at or near the maximum score of 850
    very_low_prob = logistic_regression_score([0.1], [-100], 2.0)
    assert very_low_prob["probability_of_default"] < 0.01
    assert very_low_prob["credit_score"] >= 845  # Allow small rounding differences
    assert very_low_prob["credit_score"] <= 850


def test_extreme_values():
    """Test with extreme coefficient and feature values."""
    # Very large positive log odds
    large_pos = logistic_regression_score([100], [10], 0)
    assert large_pos["log_odds"] == 1000
    assert large_pos["probability_of_default"] > 0.999
    assert large_pos["credit_score"] == 300
    
    # Very large negative log odds
    large_neg = logistic_regression_score([-100], [10], 0)
    assert large_neg["log_odds"] == -1000
    assert large_neg["probability_of_default"] < 0.001
    assert large_neg["credit_score"] == 850


def test_different_dimensions():
    """Test with coefficients and features of different dimensions."""
    # 1-dimensional
    result_1d = logistic_regression_score([0.01], [700], -2.0)
    expected_log_odds_1d = 0.01 * 700 - 2.0
    expected_prob_1d = 1 / (1 + np.exp(-expected_log_odds_1d))
    assert result_1d["log_odds"] == pytest.approx(expected_log_odds_1d)
    assert result_1d["probability_of_default"] == pytest.approx(expected_prob_1d)
    
    # 3-dimensional
    coefficients_3d = [0.01, -0.005, 0.02]
    features_3d = [700, 60, 10]
    intercept_3d = -2.5
    
    result_3d = logistic_regression_score(coefficients_3d, features_3d, intercept_3d)
    expected_log_odds_3d = 0.01 * 700 + (-0.005) * 60 + 0.02 * 10 - 2.5
    expected_prob_3d = 1 / (1 + np.exp(-expected_log_odds_3d))
    
    assert result_3d["log_odds"] == pytest.approx(expected_log_odds_3d)
    assert result_3d["probability_of_default"] == pytest.approx(expected_prob_3d)
    
    # 5-dimensional
    coefficients_5d = [0.01, -0.005, 0.02, 0.03, -0.01]
    features_5d = [700, 60, 10, 5, 20]
    intercept_5d = -2.5
    
    result_5d = logistic_regression_score(coefficients_5d, features_5d, intercept_5d)
    expected_log_odds_5d = (0.01 * 700 + (-0.005) * 60 + 0.02 * 10 + 
                           0.03 * 5 + (-0.01) * 20 - 2.5)
    expected_prob_5d = 1 / (1 + np.exp(-expected_log_odds_5d))
    
    assert result_5d["log_odds"] == pytest.approx(expected_log_odds_5d)
    assert result_5d["probability_of_default"] == pytest.approx(expected_prob_5d)


def test_numpy_arrays():
    """Test with numpy arrays instead of lists."""
    coefficients = np.array([0.01, -0.005, 0.02])
    features = np.array([700, 60, 10])
    intercept = -2.5
    
    result = logistic_regression_score(coefficients, features, intercept)
    
    expected_log_odds = 0.01 * 700 + (-0.005) * 60 + 0.02 * 10 - 2.5
    expected_prob = 1 / (1 + np.exp(-expected_log_odds))
    
    assert result["log_odds"] == pytest.approx(expected_log_odds)
    assert result["probability_of_default"] == pytest.approx(expected_prob)


def test_default_intercept():
    """Test with default intercept (0)."""
    coefficients = [0.01, -0.005, 0.02]
    features = [700, 60, 10]
    
    result = logistic_regression_score(coefficients, features)
    
    expected_log_odds = 0.01 * 700 + (-0.005) * 60 + 0.02 * 10 + 0
    expected_prob = 1 / (1 + np.exp(-expected_log_odds))
    
    assert result["log_odds"] == pytest.approx(expected_log_odds)
    assert result["probability_of_default"] == pytest.approx(expected_prob)


def test_dimension_mismatch():
    """Test error handling for dimension mismatch."""
    # More coefficients than features
    with pytest.raises(ValueError):
        logistic_regression_score([0.01, 0.02, 0.03], [700, 60])
    
    # More features than coefficients
    with pytest.raises(ValueError):
        logistic_regression_score([0.01, 0.02], [700, 60, 10])


def test_typical_credit_score_features():
    """Test with realistic credit score model features."""
    # Realistic model with typical values
    coefficients = [
        0.003,    # Income (in thousands)
        -0.01,    # Debt-to-income ratio
        -0.1,     # Number of delinquencies
        0.05,     # Length of credit history (years)
        -0.2      # Number of recent credit inquiries
    ]
    
    # Good credit profile
    good_features = [80, 0.2, 0, 10, 1]
    good_result = logistic_regression_score(coefficients, good_features, -3.0)
    
    # Average credit profile
    avg_features = [50, 0.4, 1, 5, 3]
    avg_result = logistic_regression_score(coefficients, avg_features, -1.0)
    
    # Poor credit profile
    poor_features = [30, 0.6, 3, 2, 5]
    poor_result = logistic_regression_score(coefficients, poor_features, 1.0)
    
    # Good credit should have a higher score than average, which should be higher than poor
    assert good_result["credit_score"] > avg_result["credit_score"]
    assert avg_result["credit_score"] > poor_result["credit_score"]
    
    # Check that the risk categories make sense for different profiles
    assert good_result["risk_category"] in ["Excellent", "Good"]
    assert poor_result["risk_category"] in ["Poor", "Very Poor"] 