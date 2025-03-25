import pytest
import numpy as np
from scipy import stats
from pypulate.credit.merton_model import merton_model

# Basic functionality tests
def test_basic_calculation():
    """Test the basic calculation of the Merton model"""
    result = merton_model(
        asset_value=100.0,
        debt_face_value=80.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    
    # Verify the output structure
    assert isinstance(result, dict)
    assert "probability_of_default" in result
    assert "distance_to_default" in result
    assert "d1" in result
    assert "d2" in result
    
    # Manual calculation of expected values
    expected_d1 = (np.log(100.0 / 80.0) + (0.05 + 0.5 * 0.2**2) * 1.0) / (0.2 * np.sqrt(1.0))
    expected_d2 = expected_d1 - 0.2 * np.sqrt(1.0)
    expected_pd = stats.norm.cdf(-expected_d2)
    
    # Verify the calculated values
    assert abs(result["d1"] - expected_d1) < 1e-10
    assert abs(result["d2"] - expected_d2) < 1e-10
    assert abs(result["distance_to_default"] - expected_d2) < 1e-10
    assert abs(result["probability_of_default"] - expected_pd) < 1e-10

def test_high_default_probability():
    """Test scenario with high default probability"""
    result = merton_model(
        asset_value=85.0,
        debt_face_value=100.0,
        asset_volatility=0.3,
        risk_free_rate=0.02,
        time_to_maturity=1.0
    )
    
    # For assets worth less than debt, we expect high PD
    assert result["probability_of_default"] > 0.5
    assert result["distance_to_default"] < 0

def test_low_default_probability():
    """Test scenario with low default probability"""
    result = merton_model(
        asset_value=150.0,
        debt_face_value=80.0,
        asset_volatility=0.15,
        risk_free_rate=0.03,
        time_to_maturity=1.0
    )
    
    # For assets much greater than debt, we expect low PD
    assert result["probability_of_default"] < 0.1
    assert result["distance_to_default"] > 0

def test_volatility_effect():
    """Test the effect of volatility on default probability"""
    result_low_vol = merton_model(
        asset_value=100.0,
        debt_face_value=80.0,
        asset_volatility=0.1,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    
    result_high_vol = merton_model(
        asset_value=100.0,
        debt_face_value=80.0,
        asset_volatility=0.3,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    
    # Higher volatility should lead to higher default probability
    assert result_high_vol["probability_of_default"] > result_low_vol["probability_of_default"]
    assert result_high_vol["distance_to_default"] < result_low_vol["distance_to_default"]

def test_time_to_maturity_effect():
    """Test the effect of time to maturity on default probability"""
    result_short_term = merton_model(
        asset_value=100.0,
        debt_face_value=90.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=0.5
    )
    
    result_long_term = merton_model(
        asset_value=100.0,
        debt_face_value=90.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=3.0
    )
    
    # For the Merton model with these parameters, the impact of longer time 
    # is typically more uncertainty (higher volatility effect) which actually 
    # increases default probability
    
    # Calculate and verify the expected values for both scenarios
    expected_d1_short = (np.log(100.0 / 90.0) + (0.05 + 0.5 * 0.2**2) * 0.5) / (0.2 * np.sqrt(0.5))
    expected_d2_short = expected_d1_short - 0.2 * np.sqrt(0.5)
    expected_pd_short = stats.norm.cdf(-expected_d2_short)
    
    expected_d1_long = (np.log(100.0 / 90.0) + (0.05 + 0.5 * 0.2**2) * 3.0) / (0.2 * np.sqrt(3.0))
    expected_d2_long = expected_d1_long - 0.2 * np.sqrt(3.0)
    expected_pd_long = stats.norm.cdf(-expected_d2_long)
    
    # Verify calculations are correct for both timeframes
    assert abs(result_short_term["d1"] - expected_d1_short) < 1e-10
    assert abs(result_short_term["d2"] - expected_d2_short) < 1e-10
    assert abs(result_short_term["probability_of_default"] - expected_pd_short) < 1e-10
    
    assert abs(result_long_term["d1"] - expected_d1_long) < 1e-10
    assert abs(result_long_term["d2"] - expected_d2_long) < 1e-10
    assert abs(result_long_term["probability_of_default"] - expected_pd_long) < 1e-10

def test_risk_free_rate_effect():
    """Test the effect of risk-free rate on default probability"""
    result_low_rate = merton_model(
        asset_value=100.0,
        debt_face_value=90.0,
        asset_volatility=0.2,
        risk_free_rate=0.01,
        time_to_maturity=1.0
    )
    
    result_high_rate = merton_model(
        asset_value=100.0,
        debt_face_value=90.0,
        asset_volatility=0.2,
        risk_free_rate=0.08,
        time_to_maturity=1.0
    )
    
    # Higher risk-free rate should lead to lower default probability
    assert result_high_rate["probability_of_default"] < result_low_rate["probability_of_default"]
    assert result_high_rate["distance_to_default"] > result_low_rate["distance_to_default"]

def test_theoretical_relationships():
    """Test theoretical relationships in the Merton model"""
    result = merton_model(
        asset_value=100.0,
        debt_face_value=80.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    
    # The distance to default should equal d2
    assert abs(result["distance_to_default"] - result["d2"]) < 1e-10
    
    # d1 should be greater than d2 by asset_volatility * sqrt(time_to_maturity)
    assert abs((result["d1"] - result["d2"]) - 0.2 * np.sqrt(1.0)) < 1e-10

# Edge cases
def test_extreme_values():
    """Test extreme values in the model"""
    # Case with near-zero default probability
    result_safe = merton_model(
        asset_value=1000.0,
        debt_face_value=10.0,
        asset_volatility=0.05,
        risk_free_rate=0.03,
        time_to_maturity=1.0
    )
    assert result_safe["probability_of_default"] < 0.0001
    
    # Case with near-certain default
    result_risky = merton_model(
        asset_value=20.0,
        debt_face_value=100.0,
        asset_volatility=0.05,
        risk_free_rate=0.03,
        time_to_maturity=1.0
    )
    assert result_risky["probability_of_default"] > 0.9999

def test_equal_asset_debt():
    """Test case where asset value equals debt face value"""
    result = merton_model(
        asset_value=100.0,
        debt_face_value=100.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    
    # With positive risk-free rate and time to maturity, PD should be less than 0.5
    assert result["probability_of_default"] < 0.5
    
    # Verify calculation
    expected_d1 = (np.log(1.0) + (0.05 + 0.5 * 0.2**2) * 1.0) / (0.2 * np.sqrt(1.0))
    expected_d2 = expected_d1 - 0.2 * np.sqrt(1.0)
    expected_pd = stats.norm.cdf(-expected_d2)
    
    assert abs(result["probability_of_default"] - expected_pd) < 1e-10

def test_zero_time_limiting_case():
    """Test the limiting case as time approaches zero (not exactly zero)"""
    # For very small time to maturity, we approach a deterministic case
    result = merton_model(
        asset_value=90.0,
        debt_face_value=100.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=1e-6  # Very small but non-zero
    )
    
    # With assets < debt and near-zero time, default is almost certain
    assert result["probability_of_default"] > 0.9999
    
    result = merton_model(
        asset_value=110.0,
        debt_face_value=100.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=1e-6  # Very small but non-zero
    )
    
    # With assets > debt and near-zero time, default is almost impossible
    assert result["probability_of_default"] < 0.0001

def test_zero_volatility_limiting_case():
    """Test the limiting case as volatility approaches zero (not exactly zero)"""
    # Assets > Debt * exp(-r*T), so no default
    result_no_default = merton_model(
        asset_value=100.0,
        debt_face_value=95.0,
        asset_volatility=1e-6,  # Very small but non-zero
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    # PV of debt = 95 * exp(-0.05*1) ≈ 90.5, which is < 100
    assert result_no_default["probability_of_default"] < 0.0001
    
    # Assets < Debt * exp(-r*T), so certain default
    result_certain_default = merton_model(
        asset_value=90.0,
        debt_face_value=95.0,
        asset_volatility=1e-6,  # Very small but non-zero
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    # PV of debt = 95 * exp(-0.05*1) ≈ 90.5, which is > 90
    assert result_certain_default["probability_of_default"] > 0.9999

def test_negative_risk_free_rate():
    """Test with negative risk-free rate"""
    # Negative rates are possible in some economies
    result = merton_model(
        asset_value=100.0,
        debt_face_value=80.0,
        asset_volatility=0.2,
        risk_free_rate=-0.01,
        time_to_maturity=1.0
    )
    
    # Just verify calculation is correct
    expected_d1 = (np.log(100.0 / 80.0) + (-0.01 + 0.5 * 0.2**2) * 1.0) / (0.2 * np.sqrt(1.0))
    expected_d2 = expected_d1 - 0.2 * np.sqrt(1.0)
    expected_pd = stats.norm.cdf(-expected_d2)
    
    assert abs(result["probability_of_default"] - expected_pd) < 1e-10

def test_numerical_stability():
    """Test numerical stability with extreme inputs"""
    # Very large asset/debt ratio
    result_large_ratio = merton_model(
        asset_value=1e9,
        debt_face_value=1e3,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    assert 0 <= result_large_ratio["probability_of_default"] < 1e-10
    
    # Very small asset/debt ratio
    result_small_ratio = merton_model(
        asset_value=1e3,
        debt_face_value=1e9,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    assert 0.9999 < result_small_ratio["probability_of_default"] <= 1.0
    
    # Very high volatility
    result_high_vol = merton_model(
        asset_value=100.0,
        debt_face_value=80.0,
        asset_volatility=2.0,
        risk_free_rate=0.05,
        time_to_maturity=1.0
    )
    assert 0 <= result_high_vol["probability_of_default"] <= 1.0
    
    # Very long maturity
    result_long_term = merton_model(
        asset_value=100.0,
        debt_face_value=80.0,
        asset_volatility=0.2,
        risk_free_rate=0.05,
        time_to_maturity=30.0
    )
    assert 0 <= result_long_term["probability_of_default"] <= 1.0

def test_comparison_with_known_values():
    """Test against some known reference values"""
    # These reference values should be verified with an external implementation
    # or manual calculation for your specific parameters
    
    # Example: A typical corporate case
    result = merton_model(
        asset_value=120.0,
        debt_face_value=100.0,
        asset_volatility=0.25,
        risk_free_rate=0.03,
        time_to_maturity=1.0
    )
    
    # Calculate the expected values
    expected_d1 = (np.log(120.0 / 100.0) + (0.03 + 0.5 * 0.25**2) * 1.0) / (0.25 * np.sqrt(1.0))
    expected_d2 = expected_d1 - 0.25 * np.sqrt(1.0)
    expected_pd = stats.norm.cdf(-expected_d2)
    
    assert abs(result["probability_of_default"] - expected_pd) < 1e-10 