"""
Tests for the expected_credit_loss function.
"""

import pytest
import numpy as np
from pypulate.credit.expected_credit_loss import expected_credit_loss


def test_basic_calculation():
    """Test basic ECL calculation with default time horizon and discount rate."""
    result = expected_credit_loss(pd=0.05, lgd=0.4, ead=10000)
    
    # ECL = PD × LGD × EAD × Discount Factor
    # ECL = 0.05 × 0.4 × 10000 × 1 = 200
    expected_ecl = 0.05 * 0.4 * 10000 * 1
    
    assert result["expected_credit_loss"] == expected_ecl
    assert result["expected_loss_rate"] == 0.05 * 0.4
    assert "risk_level" in result
    assert "components" in result
    assert "lifetime_ecl" in result


def test_with_discount_rate():
    """Test ECL calculation with non-zero discount rate."""
    result = expected_credit_loss(pd=0.05, lgd=0.4, ead=10000, discount_rate=0.05)
    
    # Discount factor = 1 / (1 + discount_rate) ** time_horizon
    # Discount factor = 1 / (1 + 0.05) ** 1 = 1 / 1.05 = ~0.9524
    discount_factor = 1 / (1 + 0.05) ** 1
    expected_ecl = 0.05 * 0.4 * 10000 * discount_factor
    
    assert result["expected_credit_loss"] == pytest.approx(expected_ecl)
    assert result["components"]["discount_factor"] == pytest.approx(discount_factor)


def test_with_time_horizon():
    """Test ECL calculation with different time horizon."""
    result = expected_credit_loss(pd=0.05, lgd=0.4, ead=10000, time_horizon=3.0)
    
    # Discount factor = 1 / (1 + discount_rate) ** time_horizon
    # Discount factor = 1 / (1 + 0) ** 3 = 1
    discount_factor = 1
    expected_ecl = 0.05 * 0.4 * 10000 * discount_factor
    
    # Marginal PD = 1 - (1 - PD) ** time_horizon
    # Marginal PD = 1 - (1 - 0.05) ** 3 = 1 - 0.95^3 = 1 - 0.857375 = 0.142625
    marginal_pd = 1 - (1 - 0.05) ** 3
    expected_lifetime_ecl = marginal_pd * 0.4 * 10000 * discount_factor
    
    assert result["expected_credit_loss"] == pytest.approx(expected_ecl)
    assert result["lifetime_ecl"] == pytest.approx(expected_lifetime_ecl)
    assert result["components"]["time_horizon"] == 3.0


def test_with_time_horizon_and_discount_rate():
    """Test ECL calculation with non-default time horizon and discount rate."""
    result = expected_credit_loss(pd=0.05, lgd=0.4, ead=10000, time_horizon=3.0, discount_rate=0.05)
    
    # Discount factor = 1 / (1 + discount_rate) ** time_horizon
    # Discount factor = 1 / (1 + 0.05) ** 3 = 1 / 1.157625 = ~0.8638
    discount_factor = 1 / (1 + 0.05) ** 3
    expected_ecl = 0.05 * 0.4 * 10000 * discount_factor
    
    # Marginal PD = 1 - (1 - PD) ** time_horizon
    # Marginal PD = 1 - (1 - 0.05) ** 3 = 1 - 0.95^3 = 1 - 0.857375 = 0.142625
    marginal_pd = 1 - (1 - 0.05) ** 3
    expected_lifetime_ecl = marginal_pd * 0.4 * 10000 * discount_factor
    
    assert result["expected_credit_loss"] == pytest.approx(expected_ecl)
    assert result["lifetime_ecl"] == pytest.approx(expected_lifetime_ecl)


def test_risk_levels():
    """Test different risk levels based on expected loss rate."""
    # Very Low (expected_loss_rate < 0.01)
    result_very_low = expected_credit_loss(pd=0.02, lgd=0.4, ead=10000)
    assert result_very_low["expected_loss_rate"] == pytest.approx(0.008)
    assert result_very_low["risk_level"] == "Very Low"
    
    # Low (0.01 <= expected_loss_rate < 0.03)
    result_low = expected_credit_loss(pd=0.05, lgd=0.4, ead=10000)
    assert result_low["expected_loss_rate"] == pytest.approx(0.02)
    assert result_low["risk_level"] == "Low"
    
    # Moderate (0.03 <= expected_loss_rate < 0.07)
    result_moderate = expected_credit_loss(pd=0.1, lgd=0.5, ead=10000)
    assert result_moderate["expected_loss_rate"] == pytest.approx(0.05)
    assert result_moderate["risk_level"] == "Moderate"
    
    # High (0.07 <= expected_loss_rate < 0.15)
    result_high = expected_credit_loss(pd=0.2, lgd=0.5, ead=10000)
    assert result_high["expected_loss_rate"] == pytest.approx(0.1)
    assert result_high["risk_level"] == "High"
    
    # Very High (expected_loss_rate >= 0.15)
    result_very_high = expected_credit_loss(pd=0.5, lgd=0.5, ead=10000)
    assert result_very_high["expected_loss_rate"] == pytest.approx(0.25)
    assert result_very_high["risk_level"] == "Very High"


def test_boundary_risk_levels():
    """Test risk levels at exact boundaries."""
    # At 0.01 boundary (Low)
    result_1 = expected_credit_loss(pd=0.025, lgd=0.4, ead=10000)
    assert result_1["expected_loss_rate"] == pytest.approx(0.01)
    assert result_1["risk_level"] == "Low"
    
    # At 0.03 boundary (Moderate)
    result_2 = expected_credit_loss(pd=0.06, lgd=0.5, ead=10000)
    assert result_2["expected_loss_rate"] == pytest.approx(0.03)
    assert result_2["risk_level"] == "Moderate"
    
    # At 0.07 boundary (High)
    result_3 = expected_credit_loss(pd=0.14, lgd=0.5, ead=10000)
    assert result_3["expected_loss_rate"] == pytest.approx(0.07)
    assert result_3["risk_level"] == "High"
    
    # At 0.15 boundary (Very High)
    result_4 = expected_credit_loss(pd=0.3, lgd=0.5, ead=10000)
    assert result_4["expected_loss_rate"] == pytest.approx(0.15)
    assert result_4["risk_level"] == "Very High"


def test_invalid_inputs():
    """Test invalid inputs that should raise ValueError."""
    # PD < 0
    with pytest.raises(ValueError, match="Probability of default must be between 0 and 1"):
        expected_credit_loss(pd=-0.1, lgd=0.4, ead=10000)
    
    # PD > 1
    with pytest.raises(ValueError, match="Probability of default must be between 0 and 1"):
        expected_credit_loss(pd=1.1, lgd=0.4, ead=10000)
    
    # LGD < 0
    with pytest.raises(ValueError, match="Loss given default must be between 0 and 1"):
        expected_credit_loss(pd=0.05, lgd=-0.1, ead=10000)
    
    # LGD > 1
    with pytest.raises(ValueError, match="Loss given default must be between 0 and 1"):
        expected_credit_loss(pd=0.05, lgd=1.1, ead=10000)
    
    # EAD < 0
    with pytest.raises(ValueError, match="Exposure at default must be non-negative"):
        expected_credit_loss(pd=0.05, lgd=0.4, ead=-100)


def test_edge_cases():
    """Test edge cases for ECL calculation."""
    # PD = 0 (no default risk)
    result_no_default = expected_credit_loss(pd=0, lgd=0.4, ead=10000)
    assert result_no_default["expected_credit_loss"] == 0
    assert result_no_default["lifetime_ecl"] == 0
    assert result_no_default["risk_level"] == "Very Low"
    
    # LGD = 0 (no loss if default)
    result_no_loss = expected_credit_loss(pd=0.05, lgd=0, ead=10000)
    assert result_no_loss["expected_credit_loss"] == 0
    assert result_no_loss["lifetime_ecl"] == 0
    assert result_no_loss["risk_level"] == "Very Low"
    
    # EAD = 0 (no exposure)
    result_no_exposure = expected_credit_loss(pd=0.05, lgd=0.4, ead=0)
    assert result_no_exposure["expected_credit_loss"] == 0
    assert result_no_exposure["lifetime_ecl"] == 0
    
    # PD = 1 (certain default)
    result_certain_default = expected_credit_loss(pd=1, lgd=0.4, ead=10000)
    assert result_certain_default["expected_credit_loss"] == 0.4 * 10000
    assert result_certain_default["lifetime_ecl"] == 0.4 * 10000
    assert result_certain_default["risk_level"] == "Very High"
    
    # LGD = 1 (total loss if default)
    result_total_loss = expected_credit_loss(pd=0.05, lgd=1, ead=10000)
    assert result_total_loss["expected_credit_loss"] == 0.05 * 10000
    assert result_total_loss["risk_level"] == "Moderate"


def test_components_dict():
    """Test that components dictionary contains all expected fields."""
    result = expected_credit_loss(pd=0.05, lgd=0.4, ead=10000, time_horizon=2.0, discount_rate=0.03)
    
    components = result["components"]
    assert components["probability_of_default"] == 0.05
    assert components["loss_given_default"] == 0.4
    assert components["exposure_at_default"] == 10000
    assert components["discount_factor"] == pytest.approx(1 / (1 + 0.03) ** 2)
    assert components["time_horizon"] == 2.0


def test_large_values():
    """Test with very large exposure values."""
    result = expected_credit_loss(pd=0.05, lgd=0.4, ead=1e9)
    
    expected_ecl = 0.05 * 0.4 * 1e9
    assert result["expected_credit_loss"] == pytest.approx(expected_ecl)
    assert result["expected_credit_loss"] == pytest.approx(2e7)


def test_small_values():
    """Test with very small PD and LGD values."""
    result = expected_credit_loss(pd=1e-6, lgd=1e-6, ead=10000)
    
    expected_ecl = 1e-6 * 1e-6 * 10000
    assert result["expected_credit_loss"] == expected_ecl
    assert result["risk_level"] == "Very Low" 