"""
Tests for the Altman Z-Score function.
"""

import pytest
import numpy as np
from pypulate.credit.altman_z_score import altman_z_score


def test_altman_high_risk():
    """Test with values that should result in high risk (distress zone)."""
    result = altman_z_score(
        working_capital=100000,
        retained_earnings=50000,
        ebit=75000,
        market_value_equity=200000,
        sales=500000,
        total_assets=1000000,
        total_liabilities=800000
    )
    
    assert result["z_score"] < 1.81
    assert result["zone"] == "Distress"
    assert result["risk_assessment"] == "High risk of bankruptcy"
    assert "components" in result
    assert len(result["components"]) == 5


def test_altman_grey_zone():
    """Test with values that should result in grey zone (moderate risk)."""
    result = altman_z_score(
        working_capital=150000,
        retained_earnings=120000,
        ebit=80000,
        market_value_equity=400000,
        sales=800000,
        total_assets=1000000,
        total_liabilities=600000
    )
    
    # Check that it's in the grey zone
    assert 1.81 <= result["z_score"] < 2.99
    assert result["zone"] == "Grey"
    assert result["risk_assessment"] == "Grey area, moderate risk"


def test_altman_safe_zone():
    """Test with values that should result in safe zone (low risk)."""
    result = altman_z_score(
        working_capital=400000,
        retained_earnings=600000,
        ebit=500000,
        market_value_equity=1500000,
        sales=2000000,
        total_assets=1000000,
        total_liabilities=300000
    )
    
    assert result["z_score"] >= 2.99
    assert result["zone"] == "Safe"
    assert result["risk_assessment"] == "Low risk of bankruptcy"


def test_altman_edge_case_borderline_distress():
    """Test with values right at the boundary between distress and grey zones."""
    # Calculate values that give exactly z_score = 1.81
    total_assets = 1000000
    total_liabilities = 500000
    
    # Start with some base values
    working_capital = 0.1 * total_assets  # 10% of assets
    retained_earnings = 0.1 * total_assets
    ebit = 0.1 * total_assets
    market_value_equity = 0.5 * total_assets
    sales = 0.8 * total_assets
    
    # Calculate resulting z-score
    x1 = working_capital / total_assets  # 0.1
    x2 = retained_earnings / total_assets  # 0.1
    x3 = ebit / total_assets  # 0.1
    x4 = market_value_equity / total_liabilities  # 1.0
    x5 = sales / total_assets  # 0.8
    
    z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 0.999*x5
    
    # Adjust ebit to hit the threshold
    adjustment = (1.81 - z_score) / 3.3  # 3.3 is the weight for x3
    ebit_adjusted = ebit + adjustment * total_assets
    
    result = altman_z_score(
        working_capital=working_capital,
        retained_earnings=retained_earnings,
        ebit=ebit_adjusted,
        market_value_equity=market_value_equity,
        sales=sales,
        total_assets=total_assets,
        total_liabilities=total_liabilities
    )
    
    assert np.isclose(result["z_score"], 1.81, atol=1e-2)
    assert result["zone"] == "Grey"


def test_altman_edge_case_borderline_safe():
    """Test with values right at the boundary between grey and safe zones."""
    # Calculate values that give exactly z_score = 2.99
    total_assets = 1000000
    total_liabilities = 500000
    
    # Start with some base values
    working_capital = 0.2 * total_assets
    retained_earnings = 0.2 * total_assets
    ebit = 0.2 * total_assets
    market_value_equity = 1.0 * total_assets
    sales = 1.0 * total_assets
    
    # Calculate resulting z-score
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_value_equity / total_liabilities
    x5 = sales / total_assets
    
    z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 0.999*x5
    
    # Adjust ebit to hit the threshold
    adjustment = (2.99 - z_score) / 3.3  # 3.3 is the weight for x3
    ebit_adjusted = ebit + adjustment * total_assets
    
    result = altman_z_score(
        working_capital=working_capital,
        retained_earnings=retained_earnings,
        ebit=ebit_adjusted,
        market_value_equity=market_value_equity,
        sales=sales,
        total_assets=total_assets,
        total_liabilities=total_liabilities
    )
    
    assert np.isclose(result["z_score"], 2.99, atol=1e-2)
    assert result["zone"] == "Safe"


def test_altman_zero_assets():
    """Test handling of zero total assets."""
    with pytest.raises(ZeroDivisionError):
        altman_z_score(
            working_capital=100000,
            retained_earnings=50000,
            ebit=75000,
            market_value_equity=200000,
            sales=500000,
            total_assets=0,  # Zero assets
            total_liabilities=800000
        )


def test_altman_zero_liabilities():
    """Test handling of zero total liabilities."""
    with pytest.raises(ZeroDivisionError):
        altman_z_score(
            working_capital=100000,
            retained_earnings=50000,
            ebit=75000,
            market_value_equity=200000,
            sales=500000,
            total_assets=1000000,
            total_liabilities=0  # Zero liabilities
        )


def test_altman_negative_values():
    """Test with negative financial values."""
    # Companies in distress might have negative working capital and retained earnings
    result = altman_z_score(
        working_capital=-100000,  # Negative working capital
        retained_earnings=-50000,  # Negative retained earnings
        ebit=-75000,  # Negative EBIT
        market_value_equity=200000,
        sales=500000,
        total_assets=1000000,
        total_liabilities=800000
    )
    
    # This should produce a very low z-score, well into the distress zone
    assert result["z_score"] < 1.81
    assert result["zone"] == "Distress"
    
    # Check that the components reflect the negative inputs
    assert result["components"]["x1"] < 0
    assert result["components"]["x2"] < 0
    assert result["components"]["x3"] < 0 