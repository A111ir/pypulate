import pytest
import numpy as np
from pypulate.asset.apt import apt

def test_basic_apt():
    """Test basic APT calculation with simple inputs"""
    result = apt(
        risk_free_rate=0.03,
        factor_betas=[0.8, 0.5, 0.3],
        factor_risk_premiums=[0.04, 0.02, 0.01]
    )
    assert isinstance(result, dict)
    assert abs(result['expected_return'] - 0.075) < 1e-10
    assert result['risk_free_rate'] == 0.03
    assert len(result['factor_details']) == 3

def test_single_factor():
    """Test APT with a single factor"""
    result = apt(
        risk_free_rate=0.02,
        factor_betas=[1.0],
        factor_risk_premiums=[0.03]
    )
    assert abs(result['expected_return'] - 0.05) < 1e-10
    assert result['risk_assessment'] == "Moderate risk"

def test_high_risk_scenario():
    """Test APT with high risk factors"""
    result = apt(
        risk_free_rate=0.02,
        factor_betas=[1.5, 1.2, 1.0],
        factor_risk_premiums=[0.06, 0.04, 0.03]
    )
    assert result['risk_assessment'] == "High risk"
    assert result['total_systematic_risk'] > 0.06

def test_low_risk_scenario():
    """Test APT with low risk factors"""
    result = apt(
        risk_free_rate=0.02,
        factor_betas=[0.1, 0.1, 0.1],
        factor_risk_premiums=[0.01, 0.01, 0.01]
    )
    assert result['risk_assessment'] == "Low risk"
    assert result['total_systematic_risk'] < 0.02

def test_negative_factors():
    """Test APT with negative betas and risk premiums"""
    result = apt(
        risk_free_rate=0.02,
        factor_betas=[-0.5, 0.8, -0.3],
        factor_risk_premiums=[0.04, -0.02, 0.03]
    )
    assert isinstance(result['expected_return'], float)
    for detail in result['factor_details']:
        assert 0 <= detail['contribution_pct'] <= 1

def test_zero_risk_free_rate():
    """Test APT with zero risk-free rate"""
    result = apt(
        risk_free_rate=0.0,
        factor_betas=[0.8, 0.5],
        factor_risk_premiums=[0.04, 0.02]
    )
    assert result['expected_return'] == pytest.approx(0.042)
    assert result['risk_free_rate'] == 0.0

def test_input_validation():
    """Test input validation"""
    with pytest.raises(ValueError, match="Number of factor betas must match"):
        apt(0.03, [0.8, 0.5], [0.04])
    
    with pytest.raises(ValueError, match="At least one factor must be provided"):
        apt(0.03, [], [])

def test_factor_contribution():
    """Test factor contribution calculations"""
    result = apt(
        risk_free_rate=0.03,
        factor_betas=[1.0, 0.5],
        factor_risk_premiums=[0.04, 0.02]
    )
    
    # Check factor details
    assert len(result['factor_details']) == 2
    total_contribution = sum(detail['contribution_pct'] for detail in result['factor_details'])
    assert abs(total_contribution - 1.0) < 1e-10

def test_large_number_of_factors():
    """Test APT with a large number of factors"""
    n_factors = 10
    betas = [0.5] * n_factors
    premiums = [0.02] * n_factors
    
    result = apt(
        risk_free_rate=0.03,
        factor_betas=betas,
        factor_risk_premiums=premiums
    )
    
    assert len(result['factor_details']) == n_factors
    assert isinstance(result['expected_return'], float)

def test_numerical_stability():
    """Test numerical stability with very small and large numbers"""
    result = apt(
        risk_free_rate=0.0001,
        factor_betas=[0.0001, 1000.0],
        factor_risk_premiums=[0.0001, 0.0001]
    )
    
    assert np.isfinite(result['expected_return'])
    assert np.isfinite(result['total_systematic_risk'])

def test_risk_assessment_boundaries():
    """Test risk assessment category boundaries"""
    # Test low-moderate boundary
    result_low = apt(
        risk_free_rate=0.02,
        factor_betas=[0.1],
        factor_risk_premiums=[0.19]
    )
    assert result_low['risk_assessment'] == "Low risk"
    
    result_moderate = apt(
        risk_free_rate=0.02,
        factor_betas=[0.1],
        factor_risk_premiums=[0.21]
    )
    assert result_moderate['risk_assessment'] == "Moderate risk"

def test_factor_details_structure():
    """Test the structure of factor details"""
    result = apt(
        risk_free_rate=0.03,
        factor_betas=[0.8, 0.5],
        factor_risk_premiums=[0.04, 0.02]
    )
    
    for detail in result['factor_details']:
        assert 'factor_number' in detail
        assert 'beta' in detail
        assert 'risk_premium' in detail
        assert 'contribution' in detail
        assert 'contribution_pct' in detail
        assert isinstance(detail['factor_number'], int)
        assert isinstance(detail['beta'], float)
        assert isinstance(detail['risk_premium'], float)
        assert isinstance(detail['contribution'], float)
        assert isinstance(detail['contribution_pct'], float)

def test_expected_return_components():
    """Test that expected return components sum correctly"""
    risk_free_rate = 0.03
    factor_betas = [0.8, 0.5]
    factor_risk_premiums = [0.04, 0.02]
    
    result = apt(
        risk_free_rate=risk_free_rate,
        factor_betas=factor_betas,
        factor_risk_premiums=factor_risk_premiums
    )
    
    # Calculate expected return manually
    manual_factor_returns = sum(beta * premium 
                              for beta, premium in zip(factor_betas, factor_risk_premiums))
    manual_expected_return = risk_free_rate + manual_factor_returns
    
    assert abs(result['expected_return'] - manual_expected_return) < 1e-10

def test_documentation_example():
    """Test the example from the function's docstring"""
    result = apt(
        risk_free_rate=0.03,
        factor_betas=[0.8, 0.5, 0.3],
        factor_risk_premiums=[0.04, 0.02, 0.01]
    )
    assert abs(result['expected_return'] - 0.075) < 1e-10 