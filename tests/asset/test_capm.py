"""
Tests for Capital Asset Pricing Model (CAPM) implementation.
"""

import pytest
import numpy as np
from pypulate.asset import capm


def test_basic_capm():
    """Test basic CAPM calculation."""
    result = capm(
        risk_free_rate=0.03,  # 3% risk-free rate
        beta=1.0,             # Market beta
        market_return=0.08    # 8% market return
    )
    
    # Expected return should be: 0.03 + 1.0 * (0.08 - 0.03) = 0.08
    assert abs(result['expected_return'] - 0.08) < 1e-10
    assert result['market_risk_premium'] == 0.05  # 8% - 3%
    assert result['risk_assessment'] == "Market-level risk"  # beta = 1.0


def test_vectorized_capm():
    """Test vectorized CAPM calculation for multiple assets."""
    result = capm(
        risk_free_rate=0.03,
        beta=[0.8, 1.0, 1.2, 1.5, 1.8],
        market_return=0.08
    )
    
    assert len(result) == 5  # Should return list of 5 results
    
    # Check expected returns for each beta
    expected_returns = [
        0.03 + 0.8 * (0.08 - 0.03),  # β = 0.8
        0.03 + 1.0 * (0.08 - 0.03),  # β = 1.0
        0.03 + 1.2 * (0.08 - 0.03),  # β = 1.2
        0.03 + 1.5 * (0.08 - 0.03),  # β = 1.5
        0.03 + 1.8 * (0.08 - 0.03)   # β = 1.8
    ]
    
    for r, er in zip(result, expected_returns):
        assert abs(r['expected_return'] - er) < 1e-10
    
    # Check risk assessments
    assert result[0]['risk_assessment'] == "Below-market risk"  # β = 0.8
    assert result[1]['risk_assessment'] == "Market-level risk"  # β = 1.0
    assert result[2]['risk_assessment'] == "Above-market risk"  # β = 1.2
    assert result[3]['risk_assessment'] == "High risk"         # β = 1.5
    assert result[4]['risk_assessment'] == "High risk"         # β = 1.8


def test_numpy_array_input():
    """Test CAPM with NumPy array inputs."""
    betas = np.array([0.5, 1.0, 1.5])
    result = capm(
        risk_free_rate=0.03,
        beta=betas,
        market_return=0.08
    )
    
    assert len(result) == 3
    assert result[0]['risk_assessment'] == "Low risk"
    assert result[1]['risk_assessment'] == "Market-level risk"
    assert result[2]['risk_assessment'] == "High risk"


def test_broadcasting():
    """Test broadcasting with different input shapes."""
    result = capm(
        risk_free_rate=[0.02, 0.03, 0.04],
        beta=1.0,
        market_return=0.08
    )
    
    assert len(result) == 3
    for r, rf in zip(result, [0.02, 0.03, 0.04]):
        expected = rf + 1.0 * (0.08 - rf)
        assert abs(r['expected_return'] - expected) < 1e-10


def test_input_validation():
    """Test input validation."""
    # Test negative risk-free rate
    with pytest.raises(ValueError, match="Risk-free rate cannot be negative"):
        capm(
            risk_free_rate=-0.01,
            beta=1.0,
            market_return=0.08
        )
    
    # Test that negative market premium is allowed
    result = capm(
        risk_free_rate=0.06,
        beta=1.0,
        market_return=0.04
    )
    assert result['market_risk_premium'] < 0  # Should allow negative market premium


def test_high_risk_stock():
    """Test CAPM for high-risk stock (beta > 1.5)."""
    result = capm(
        risk_free_rate=0.03,
        beta=1.8,             # High beta
        market_return=0.08
    )
    
    # Expected return should be: 0.03 + 1.8 * (0.08 - 0.03) = 0.12
    assert abs(result['expected_return'] - 0.12) < 1e-10
    assert result['risk_assessment'] == "High risk"


def test_low_risk_stock():
    """Test CAPM for low-risk stock (beta < 0.8)."""
    result = capm(
        risk_free_rate=0.03,
        beta=0.5,             # Low beta
        market_return=0.08
    )
    
    # Expected return should be: 0.03 + 0.5 * (0.08 - 0.03) = 0.055
    assert abs(result['expected_return'] - 0.055) < 1e-10
    assert result['risk_assessment'] == "Low risk"


def test_defensive_stock():
    """Test CAPM for defensive stock (0.8 ≤ beta < 1.0)."""
    result = capm(
        risk_free_rate=0.03,
        beta=0.9,             # Defensive beta
        market_return=0.08
    )
    
    # Expected return should be: 0.03 + 0.9 * (0.08 - 0.03) = 0.075
    assert abs(result['expected_return'] - 0.075) < 1e-10
    assert result['risk_assessment'] == "Below-market risk"


def test_aggressive_stock():
    """Test CAPM for aggressive stock (1.2 ≤ beta < 1.5)."""
    result = capm(
        risk_free_rate=0.03,
        beta=1.3,             # Aggressive beta
        market_return=0.08
    )
    
    # Expected return should be: 0.03 + 1.3 * (0.08 - 0.03) = 0.095
    assert abs(result['expected_return'] - 0.095) < 1e-10
    assert result['risk_assessment'] == "Above-market risk"


def test_negative_market_premium():
    """Test CAPM with negative market risk premium."""
    result = capm(
        risk_free_rate=0.06,  # Higher risk-free rate
        beta=1.0,
        market_return=0.04    # Lower market return
    )
    
    # Expected return should be: 0.06 + 1.0 * (0.04 - 0.06) = 0.04
    assert abs(result['expected_return'] - 0.04) < 1e-10
    assert result['market_risk_premium'] < 0


def test_zero_beta():
    """Test CAPM with zero beta (risk-free asset)."""
    result = capm(
        risk_free_rate=0.03,
        beta=0.0,             # Risk-free beta
        market_return=0.08
    )
    
    # Expected return should equal risk-free rate
    assert abs(result['expected_return'] - 0.03) < 1e-10
    assert result['risk_assessment'] == "Low risk"


def test_negative_beta():
    """Test CAPM with negative beta (inverse relationship with market)."""
    result = capm(
        risk_free_rate=0.03,
        beta=-0.5,            # Negative beta
        market_return=0.08
    )
    
    # Expected return should be: 0.03 + (-0.5) * (0.08 - 0.03) = 0.005
    assert abs(result['expected_return'] - 0.005) < 1e-10
    assert result['risk_assessment'] == "Low risk"


def test_extreme_values():
    """Test CAPM with extreme values."""
    result = capm(
        risk_free_rate=0.10,  # High risk-free rate
        beta=2.5,             # Very high beta
        market_return=0.20    # High market return
    )
    
    # Expected return should be: 0.10 + 2.5 * (0.20 - 0.10) = 0.35
    assert abs(result['expected_return'] - 0.35) < 1e-10
    assert result['risk_assessment'] == "High risk"


def test_numerical_precision():
    """Test CAPM with small values to check numerical precision."""
    result = capm(
        risk_free_rate=0.001,  # Very small risk-free rate
        beta=1.0,
        market_return=0.002    # Very small market return
    )
    
    # Expected return should be: 0.001 + 1.0 * (0.002 - 0.001) = 0.002
    assert abs(result['expected_return'] - 0.002) < 1e-10
    assert abs(result['market_risk_premium'] - 0.001) < 1e-10


def test_return_components():
    """Test that CAPM returns all required components."""
    result = capm(
        risk_free_rate=0.03,
        beta=1.0,
        market_return=0.08
    )
    
    # Check that all components are present
    assert 'expected_return' in result
    assert 'risk_free_rate' in result
    assert 'beta' in result
    assert 'market_return' in result
    assert 'market_risk_premium' in result
    assert 'risk_assessment' in result
    
    # Check that values match inputs
    assert result['risk_free_rate'] == 0.03
    assert result['beta'] == 1.0
    assert result['market_return'] == 0.08 