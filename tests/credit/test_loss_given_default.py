import pytest
import numpy as np
from pypulate.credit.loss_given_default import loss_given_default

# Basic functionality tests
def test_basic_calculation():
    """Test basic LGD calculation with collateral covering loan"""
    result = loss_given_default(collateral_value=100000, loan_amount=80000)
    assert isinstance(result, dict)
    assert "lgd" in result
    assert "present_value_lgd" in result
    assert "risk_level" in result
    assert "components" in result
    assert result["lgd"] == 0.0  # Collateral fully covers loan after liquidation costs
    assert result["risk_level"] == "Very Low"
    assert abs(result["present_value_lgd"] - 0.0) < 1e-10

def test_insufficient_collateral():
    """Test LGD calculation when collateral is insufficient"""
    result = loss_given_default(collateral_value=50000, loan_amount=80000)
    assert result["lgd"] > 0
    assert result["lgd"] < 1
    expected_lgd = 1 - (50000 * 0.9 / 80000)  # (1 - net_collateral/loan)
    assert abs(result["lgd"] - expected_lgd) < 1e-10
    assert result["risk_level"] in ["Moderate", "High"]  # Depending on exact LGD value

def test_with_recovery_rate():
    """Test LGD calculation with historical recovery rate"""
    result = loss_given_default(
        collateral_value=100000, 
        loan_amount=60000, 
        recovery_rate=0.7
    )
    # LTV is 0.6, so weight_collateral should be 0.6
    # collateral_lgd is 0 (full coverage)
    # Weighted LGD = (0 * 0.6) + ((1-0.7) * 0.4) = 0.12
    assert abs(result["lgd"] - 0.12) < 1e-10
    assert result["risk_level"] == "Low"
    assert "recovery_rate" in result["components"]
    assert "weight_collateral" in result["components"]
    assert "weight_historical" in result["components"]

def test_no_collateral():
    """Test LGD calculation with no collateral"""
    result = loss_given_default(collateral_value=0, loan_amount=10000)
    assert result["lgd"] == 1.0
    assert result["risk_level"] == "Very High"
    assert result["components"]["loan_to_value"] == float('inf')

def test_present_value_calculation():
    """Test the present value calculation of LGD"""
    result = loss_given_default(
        collateral_value=50000, 
        loan_amount=80000,
        time_to_recovery=2.0
    )
    # Expected LGD = 1 - (50000 * 0.9 / 80000) = 0.4375
    expected_lgd = 1 - (50000 * 0.9 / 80000)
    # PV factor = 1 / (1 + 0.05)^2 = 0.9070294784
    expected_pv_factor = 1 / ((1 + 0.05) ** 2)
    expected_pv_lgd = expected_lgd * expected_pv_factor
    
    assert abs(result["lgd"] - expected_lgd) < 1e-10
    assert abs(result["components"]["time_value_factor"] - expected_pv_factor) < 1e-10
    assert abs(result["present_value_lgd"] - expected_pv_lgd) < 1e-10

# Risk level tests
def test_very_low_risk():
    """Test very low risk classification"""
    result = loss_given_default(collateral_value=200000, loan_amount=100000)
    assert result["lgd"] < 0.1
    assert result["risk_level"] == "Very Low"

def test_low_risk():
    """Test low risk classification"""
    # Create a scenario with LGD between 0.1 and 0.3
    result = loss_given_default(
        collateral_value=100000, 
        loan_amount=80000, 
        recovery_rate=0.6
    )
    assert 0.1 <= result["lgd"] < 0.3
    assert result["risk_level"] == "Low"

def test_moderate_risk():
    """Test moderate risk classification"""
    # Create a scenario with LGD between 0.3 and 0.5
    result = loss_given_default(
        collateral_value=65000, 
        loan_amount=100000
    )
    assert 0.3 <= result["lgd"] < 0.5
    assert result["risk_level"] == "Moderate"

def test_high_risk():
    """Test high risk classification"""
    # Create a scenario with LGD between 0.5 and 0.7
    result = loss_given_default(
        collateral_value=40000, 
        loan_amount=100000
    )
    assert 0.5 <= result["lgd"] < 0.7
    assert result["risk_level"] == "High"

def test_very_high_risk():
    """Test very high risk classification"""
    # Create a scenario with LGD >= 0.7
    result = loss_given_default(
        collateral_value=25000, 
        loan_amount=100000
    )
    assert result["lgd"] >= 0.7
    assert result["risk_level"] == "Very High"

# Parameter validation tests
def test_negative_loan_amount():
    """Test validation for negative loan amount"""
    with pytest.raises(ValueError, match="Loan amount must be positive"):
        loss_given_default(collateral_value=100000, loan_amount=-50000)

def test_zero_loan_amount():
    """Test validation for zero loan amount"""
    with pytest.raises(ValueError, match="Loan amount must be positive"):
        loss_given_default(collateral_value=100000, loan_amount=0)

def test_negative_collateral():
    """Test validation for negative collateral"""
    with pytest.raises(ValueError, match="Collateral value cannot be negative"):
        loss_given_default(collateral_value=-10000, loan_amount=50000)

def test_invalid_recovery_rate():
    """Test validation for invalid recovery rate"""
    with pytest.raises(ValueError, match="Recovery rate must be between 0 and 1"):
        loss_given_default(collateral_value=100000, loan_amount=50000, recovery_rate=1.2)
    
    with pytest.raises(ValueError, match="Recovery rate must be between 0 and 1"):
        loss_given_default(collateral_value=100000, loan_amount=50000, recovery_rate=-0.1)

def test_invalid_liquidation_costs():
    """Test validation for invalid liquidation costs"""
    with pytest.raises(ValueError, match="Liquidation costs must be between 0 and 1"):
        loss_given_default(collateral_value=100000, loan_amount=50000, liquidation_costs=1.2)
    
    with pytest.raises(ValueError, match="Liquidation costs must be between 0 and 1"):
        loss_given_default(collateral_value=100000, loan_amount=50000, liquidation_costs=-0.1)

def test_invalid_time_to_recovery():
    """Test validation for invalid time to recovery"""
    with pytest.raises(ValueError, match="Time to recovery must be positive"):
        loss_given_default(collateral_value=100000, loan_amount=50000, time_to_recovery=0)
    
    with pytest.raises(ValueError, match="Time to recovery must be positive"):
        loss_given_default(collateral_value=100000, loan_amount=50000, time_to_recovery=-1)

# Edge case tests
def test_barely_covered_loan():
    """Test scenario where loan is barely covered by collateral"""
    result = loss_given_default(
        collateral_value=100000, 
        loan_amount=90000
    )
    # Net collateral = 100000 * 0.9 = 90000, which equals loan amount
    assert result["lgd"] == 0.0
    assert result["risk_level"] == "Very Low"

def test_barely_insufficient_collateral():
    """Test scenario where collateral is barely insufficient"""
    result = loss_given_default(
        collateral_value=100000, 
        loan_amount=90001
    )
    # Net collateral = 100000 * 0.9 = 90000, which is 1 less than loan amount
    expected_lgd = 1 - (90000 / 90001)
    assert abs(result["lgd"] - expected_lgd) < 1e-10

def test_different_liquidation_costs():
    """Test impact of different liquidation costs"""
    # Update values to ensure we get different LGD values
    result1 = loss_given_default(
        collateral_value=80000, 
        loan_amount=100000,
        liquidation_costs=0.1  # default
    )
    
    result2 = loss_given_default(
        collateral_value=80000, 
        loan_amount=100000,
        liquidation_costs=0.2
    )
    
    # Higher liquidation costs should result in higher LGD
    assert result2["lgd"] > result1["lgd"]

def test_different_recovery_times():
    """Test impact of different recovery times"""
    result1 = loss_given_default(
        collateral_value=70000, 
        loan_amount=100000,
        time_to_recovery=1.0  # default
    )
    
    result2 = loss_given_default(
        collateral_value=70000, 
        loan_amount=100000,
        time_to_recovery=3.0
    )
    
    # Same LGD but different present values
    assert abs(result1["lgd"] - result2["lgd"]) < 1e-10
    assert result1["present_value_lgd"] > result2["present_value_lgd"]  # Longer time means lower PV

def test_ltv_thresholds():
    """Test LTV threshold effects on weighting when recovery rate is provided"""
    # LTV < 0.5 (weight_collateral = 0.8)
    result1 = loss_given_default(
        collateral_value=100000, 
        loan_amount=45000,
        recovery_rate=0.5
    )
    
    # 0.5 < LTV < 0.8 (weight_collateral = 0.6)
    result2 = loss_given_default(
        collateral_value=100000, 
        loan_amount=65000,
        recovery_rate=0.5
    )
    
    # LTV > 0.8 (weight_collateral = 0.4)
    result3 = loss_given_default(
        collateral_value=100000, 
        loan_amount=85000,
        recovery_rate=0.5
    )
    
    # Use approximate equality for floating point comparisons
    assert abs(result1["components"]["weight_collateral"] - 0.8) < 1e-10
    assert abs(result1["components"]["weight_historical"] - 0.2) < 1e-10
    
    assert abs(result2["components"]["weight_collateral"] - 0.6) < 1e-10
    assert abs(result2["components"]["weight_historical"] - 0.4) < 1e-10
    
    assert abs(result3["components"]["weight_collateral"] - 0.4) < 1e-10
    assert abs(result3["components"]["weight_historical"] - 0.6) < 1e-10

def test_exact_ltv_boundary():
    """Test behavior at exact LTV boundaries"""
    # LTV = 0.5 exactly
    result1 = loss_given_default(
        collateral_value=100000, 
        loan_amount=50000,
        recovery_rate=0.5
    )
    
    # LTV = 0.8 exactly
    result2 = loss_given_default(
        collateral_value=100000, 
        loan_amount=80000,
        recovery_rate=0.5
    )
    
    assert abs(result1["components"]["weight_collateral"] - 0.8) < 1e-10  # <= 0.5
    assert abs(result2["components"]["weight_collateral"] - 0.6) < 1e-10  # <= 0.8 but > 0.5 