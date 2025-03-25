"""
Tests for the exposure_at_default function.
"""

import pytest
import numpy as np
from pypulate.credit.exposure_at_default import exposure_at_default


def test_basic_calculation():
    """Test basic EAD calculation with default credit conversion factor."""
    result = exposure_at_default(current_balance=100000, undrawn_amount=50000)
    
    # EAD = current_balance + (undrawn_amount * credit_conversion_factor)
    # EAD = 100000 + (50000 * 0.5) = 125000
    expected_ead = 100000 + (50000 * 0.5)
    
    assert result["ead"] == expected_ead
    assert "risk_level" in result
    assert "components" in result
    assert "regulatory_ead" in result
    assert "stressed_ead" in result


def test_custom_ccf():
    """Test EAD calculation with custom credit conversion factor."""
    result = exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=0.7)
    
    # EAD = current_balance + (undrawn_amount * credit_conversion_factor)
    # EAD = 100000 + (50000 * 0.7) = 135000
    expected_ead = 100000 + (50000 * 0.7)
    
    assert result["ead"] == expected_ead
    assert result["components"]["credit_conversion_factor"] == 0.7


def test_utilization_rates():
    """Test different utilization rates and resulting risk levels."""
    # Low utilization (< 0.3)
    result_low = exposure_at_default(current_balance=10000, undrawn_amount=90000)
    assert result_low["components"]["utilization_rate"] == 0.1
    assert result_low["risk_level"] == "Low"
    assert result_low["components"]["regulatory_ccf"] == 0.2
    
    # Moderate utilization (0.3 <= rate < 0.7)
    result_moderate = exposure_at_default(current_balance=40000, undrawn_amount=60000)
    assert result_moderate["components"]["utilization_rate"] == 0.4
    assert result_moderate["risk_level"] == "Moderate"
    assert result_moderate["components"]["regulatory_ccf"] == 0.4
    
    # High utilization (>= 0.7)
    result_high = exposure_at_default(current_balance=80000, undrawn_amount=20000)
    assert result_high["components"]["utilization_rate"] == 0.8
    assert result_high["risk_level"] == "High"
    assert result_high["components"]["regulatory_ccf"] == 0.8


def test_regulatory_ccf_brackets():
    """Test all regulatory CCF brackets based on utilization rate."""
    # Utilization < 0.3: CCF = 0.2
    result_1 = exposure_at_default(current_balance=20000, undrawn_amount=80000)
    assert result_1["components"]["utilization_rate"] == 0.2
    assert result_1["components"]["regulatory_ccf"] == 0.2
    
    # 0.3 <= Utilization < 0.5: CCF = 0.4
    result_2 = exposure_at_default(current_balance=40000, undrawn_amount=60000)
    assert result_2["components"]["utilization_rate"] == 0.4
    assert result_2["components"]["regulatory_ccf"] == 0.4
    
    # 0.5 <= Utilization < 0.7: CCF = 0.6
    result_3 = exposure_at_default(current_balance=60000, undrawn_amount=40000)
    assert result_3["components"]["utilization_rate"] == 0.6
    assert result_3["components"]["regulatory_ccf"] == 0.6
    
    # 0.7 <= Utilization < 0.9: CCF = 0.8
    result_4 = exposure_at_default(current_balance=80000, undrawn_amount=20000)
    assert result_4["components"]["utilization_rate"] == 0.8
    assert result_4["components"]["regulatory_ccf"] == 0.8
    
    # Utilization >= 0.9: CCF = 1.0
    result_5 = exposure_at_default(current_balance=95000, undrawn_amount=5000)
    assert result_5["components"]["utilization_rate"] == 0.95
    assert result_5["components"]["regulatory_ccf"] == 1.0


def test_regulatory_ead():
    """Test regulatory EAD calculation."""
    result = exposure_at_default(current_balance=30000, undrawn_amount=70000)
    
    # Utilization rate = 30000 / (30000 + 70000) = 0.3
    # For utilization rate 0.3, regulatory_ccf = 0.4
    # Regulatory EAD = current_balance + (undrawn_amount * regulatory_ccf)
    # Regulatory EAD = 30000 + (70000 * 0.4) = 58000
    expected_regulatory_ead = 30000 + (70000 * 0.4)
    
    assert result["regulatory_ead"] == expected_regulatory_ead


def test_stressed_ead():
    """Test stressed EAD calculation."""
    result = exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=0.6)
    
    # Stress CCF = min(1.0, credit_conversion_factor * 1.5)
    # Stress CCF = min(1.0, 0.6 * 1.5) = min(1.0, 0.9) = 0.9
    # Stressed EAD = current_balance + (undrawn_amount * stress_ccf)
    # Stressed EAD = 100000 + (50000 * 0.9) = 145000
    expected_stress_ccf = min(1.0, 0.6 * 1.5)
    expected_stressed_ead = 100000 + (50000 * expected_stress_ccf)
    
    assert result["components"]["stress_ccf"] == expected_stress_ccf
    assert result["stressed_ead"] == expected_stressed_ead


def test_ead_percentage():
    """Test EAD percentage calculation."""
    result = exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=0.5)
    
    # Total facility = 100000 + 50000 = 150000
    # EAD = 100000 + (50000 * 0.5) = 125000
    # EAD percentage = 125000 / 150000 = 0.8333...
    expected_total_facility = 100000 + 50000
    expected_ead = 100000 + (50000 * 0.5)
    expected_ead_percentage = expected_ead / expected_total_facility
    
    assert result["components"]["total_facility"] == expected_total_facility
    assert result["ead"] == expected_ead
    assert result["ead_percentage"] == pytest.approx(expected_ead_percentage)


def test_invalid_inputs():
    """Test invalid inputs that should raise ValueError."""
    # Negative current balance
    with pytest.raises(ValueError, match="Current balance cannot be negative"):
        exposure_at_default(current_balance=-10000, undrawn_amount=50000)
    
    # Negative undrawn amount
    with pytest.raises(ValueError, match="Undrawn amount cannot be negative"):
        exposure_at_default(current_balance=100000, undrawn_amount=-10000)
    
    # CCF < 0
    with pytest.raises(ValueError, match="Credit conversion factor must be between 0 and 1"):
        exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=-0.1)
    
    # CCF > 1
    with pytest.raises(ValueError, match="Credit conversion factor must be between 0 and 1"):
        exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=1.1)


def test_edge_cases():
    """Test edge cases for EAD calculation."""
    # Zero current balance
    result_zero_balance = exposure_at_default(current_balance=0, undrawn_amount=100000)
    assert result_zero_balance["ead"] == 50000  # 0 + (100000 * 0.5)
    assert result_zero_balance["components"]["utilization_rate"] == 0
    assert result_zero_balance["risk_level"] == "Low"
    
    # Zero undrawn amount
    result_zero_undrawn = exposure_at_default(current_balance=100000, undrawn_amount=0)
    assert result_zero_undrawn["ead"] == 100000  # 100000 + (0 * 0.5)
    assert result_zero_undrawn["components"]["utilization_rate"] == 1.0
    assert result_zero_undrawn["risk_level"] == "High"
    
    # Zero total facility
    result_zero_total = exposure_at_default(current_balance=0, undrawn_amount=0)
    assert result_zero_total["ead"] == 0
    assert result_zero_total["components"]["utilization_rate"] == 0
    assert result_zero_total["ead_percentage"] == 0
    
    # CCF = 0
    result_zero_ccf = exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=0)
    assert result_zero_ccf["ead"] == 100000  # 100000 + (50000 * 0)
    assert result_zero_ccf["components"]["stress_ccf"] == 0  # min(1.0, 0 * 1.5)
    
    # CCF = 1
    result_one_ccf = exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=1)
    assert result_one_ccf["ead"] == 150000  # 100000 + (50000 * 1)
    assert result_one_ccf["components"]["stress_ccf"] == 1  # min(1.0, 1 * 1.5)
    assert result_one_ccf["stressed_ead"] == 150000  # Same as EAD when stress_ccf is capped at 1


def test_large_values():
    """Test with very large values to ensure numerical stability."""
    result = exposure_at_default(current_balance=1e9, undrawn_amount=5e8)
    
    expected_ead = 1e9 + (5e8 * 0.5)
    assert result["ead"] == expected_ead
    assert result["ead"] == 1.25e9


def test_small_values():
    """Test with very small values to ensure numerical stability."""
    result = exposure_at_default(current_balance=1.0, undrawn_amount=1.0)
    
    expected_ead = 1.0 + (1.0 * 0.5)
    assert result["ead"] == expected_ead
    assert result["ead"] == 1.5


def test_components_dict():
    """Test that components dictionary contains all expected fields."""
    result = exposure_at_default(current_balance=100000, undrawn_amount=50000, credit_conversion_factor=0.6)
    
    components = result["components"]
    assert components["current_balance"] == 100000
    assert components["undrawn_amount"] == 50000
    assert components["total_facility"] == 150000
    assert components["utilization_rate"] == pytest.approx(2/3)
    assert components["credit_conversion_factor"] == 0.6
    assert "regulatory_ccf" in components
    assert "stress_ccf" in components 