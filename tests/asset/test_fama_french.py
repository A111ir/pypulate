"""
Tests for Fama-French factor models implementation.
"""

import pytest
import numpy as np
from pypulate.asset import fama_french_three_factor, fama_french_five_factor


def test_basic_three_factor():
    """Test basic Fama-French three-factor calculation."""
    result = fama_french_three_factor(
        risk_free_rate=0.03,    # 3% risk-free rate
        market_beta=1.2,        # Market beta
        size_beta=0.5,          # Size beta
        value_beta=0.3,         # Value beta
        market_premium=0.05,    # Market premium
        size_premium=0.02,      # Size premium
        value_premium=0.03      # Value premium
    )
    
    # Expected return: 0.03 + 1.2*0.05 + 0.5*0.02 + 0.3*0.03 = 0.109
    assert abs(result['expected_return'] - 0.109) < 1e-10
    assert result['risk_assessment'] == "Above-average risk"


def test_three_factor_risk_levels():
    """Test different risk levels in three-factor model."""
    # Low risk case
    low_risk = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=0.5,
        size_beta=0.1,
        value_beta=0.1,
        market_premium=0.02,
        size_premium=0.01,
        value_premium=0.01
    )
    assert low_risk['risk_assessment'] == "Low risk"
    
    # Moderate risk case
    moderate_risk = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.3,
        value_beta=0.2,
        market_premium=0.04,
        size_premium=0.02,
        value_premium=0.02
    )
    assert moderate_risk['risk_assessment'] == "Moderate risk"
    
    # High risk case
    high_risk = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=1.5,
        size_beta=0.8,
        value_beta=0.7,
        market_premium=0.06,
        size_premium=0.04,
        value_premium=0.04
    )
    assert high_risk['risk_assessment'] == "High risk"


def test_three_factor_contributions():
    """Test factor contributions in three-factor model."""
    result = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.5,
        value_beta=0.5,
        market_premium=0.06,
        size_premium=0.02,
        value_premium=0.02
    )
    
    # Check factor contributions
    market_contrib = result['factor_contributions']['market']['contribution']
    size_contrib = result['factor_contributions']['size']['contribution']
    value_contrib = result['factor_contributions']['value']['contribution']
    
    assert abs(market_contrib - 0.06) < 1e-10  # 1.0 * 0.06
    assert abs(size_contrib - 0.01) < 1e-10    # 0.5 * 0.02
    assert abs(value_contrib - 0.01) < 1e-10    # 0.5 * 0.02


def test_basic_five_factor():
    """Test basic Fama-French five-factor calculation."""
    result = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=1.2,
        size_beta=0.5,
        value_beta=0.3,
        profitability_beta=0.2,
        investment_beta=0.1,
        market_premium=0.05,
        size_premium=0.02,
        value_premium=0.03,
        profitability_premium=0.01,
        investment_premium=0.01
    )
    
    # Expected return: 0.03 + 1.2*0.05 + 0.5*0.02 + 0.3*0.03 + 0.2*0.01 + 0.1*0.01 = 0.112
    assert abs(result['expected_return'] - 0.112) < 1e-10
    assert result['risk_assessment'] == "Above-average risk"


def test_five_factor_risk_levels():
    """Test different risk levels in five-factor model."""
    # Low risk case
    low_risk = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=0.5,
        size_beta=0.1,
        value_beta=0.1,
        profitability_beta=0.1,
        investment_beta=0.1,
        market_premium=0.02,
        size_premium=0.01,
        value_premium=0.01,
        profitability_premium=0.01,
        investment_premium=0.01
    )
    assert low_risk['risk_assessment'] == "Low risk"
    
    # Moderate risk case
    moderate_risk = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.2,
        value_beta=0.2,
        profitability_beta=0.2,
        investment_beta=0.2,
        market_premium=0.04,
        size_premium=0.02,
        value_premium=0.02,
        profitability_premium=0.02,
        investment_premium=0.02
    )
    assert moderate_risk['risk_assessment'] == "Moderate risk"
    
    # High risk case
    high_risk = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=1.5,
        size_beta=0.5,
        value_beta=0.5,
        profitability_beta=0.5,
        investment_beta=0.5,
        market_premium=0.06,
        size_premium=0.03,
        value_premium=0.03,
        profitability_premium=0.03,
        investment_premium=0.03
    )
    assert high_risk['risk_assessment'] == "High risk"


def test_five_factor_contributions():
    """Test factor contributions in five-factor model."""
    result = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.5,
        value_beta=0.5,
        profitability_beta=0.5,
        investment_beta=0.5,
        market_premium=0.06,
        size_premium=0.02,
        value_premium=0.02,
        profitability_premium=0.02,
        investment_premium=0.02
    )
    
    # Check factor contributions
    market_contrib = result['factor_contributions']['market']['contribution']
    size_contrib = result['factor_contributions']['size']['contribution']
    value_contrib = result['factor_contributions']['value']['contribution']
    prof_contrib = result['factor_contributions']['profitability']['contribution']
    inv_contrib = result['factor_contributions']['investment']['contribution']
    
    assert abs(market_contrib - 0.06) < 1e-10  # 1.0 * 0.06
    assert abs(size_contrib - 0.01) < 1e-10    # 0.5 * 0.02
    assert abs(value_contrib - 0.01) < 1e-10   # 0.5 * 0.02
    assert abs(prof_contrib - 0.01) < 1e-10    # 0.5 * 0.02
    assert abs(inv_contrib - 0.01) < 1e-10     # 0.5 * 0.02


def test_contribution_percentages():
    """Test that contribution percentages sum to 1."""
    # Three-factor model
    three_factor = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.5,
        value_beta=0.5,
        market_premium=0.06,
        size_premium=0.02,
        value_premium=0.02
    )
    
    three_factor_pcts = [
        three_factor['factor_contributions'][factor]['contribution_pct']
        for factor in ['market', 'size', 'value']
    ]
    assert abs(sum(three_factor_pcts) - 1.0) < 1e-10
    
    # Five-factor model
    five_factor = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.5,
        value_beta=0.5,
        profitability_beta=0.5,
        investment_beta=0.5,
        market_premium=0.06,
        size_premium=0.02,
        value_premium=0.02,
        profitability_premium=0.02,
        investment_premium=0.02
    )
    
    five_factor_pcts = [
        five_factor['factor_contributions'][factor]['contribution_pct']
        for factor in ['market', 'size', 'value', 'profitability', 'investment']
    ]
    assert abs(sum(five_factor_pcts) - 1.0) < 1e-10


def test_negative_premiums():
    """Test models with negative risk premiums."""
    # Three-factor model with negative premiums
    three_factor = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.5,
        value_beta=0.5,
        market_premium=-0.02,
        size_premium=-0.01,
        value_premium=-0.01
    )
    assert three_factor['expected_return'] < three_factor['risk_free_rate']
    
    # Five-factor model with negative premiums
    five_factor = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=1.0,
        size_beta=0.5,
        value_beta=0.5,
        profitability_beta=0.5,
        investment_beta=0.5,
        market_premium=-0.02,
        size_premium=-0.01,
        value_premium=-0.01,
        profitability_premium=-0.01,
        investment_premium=-0.01
    )
    assert five_factor['expected_return'] < five_factor['risk_free_rate']


def test_zero_betas():
    """Test models with zero betas."""
    # Three-factor model with zero betas
    three_factor = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=0.0,
        size_beta=0.0,
        value_beta=0.0,
        market_premium=0.05,
        size_premium=0.02,
        value_premium=0.02
    )
    assert abs(three_factor['expected_return'] - 0.03) < 1e-10  # Should equal risk-free rate
    assert three_factor['risk_assessment'] == "Low risk"
    
    # Five-factor model with zero betas
    five_factor = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=0.0,
        size_beta=0.0,
        value_beta=0.0,
        profitability_beta=0.0,
        investment_beta=0.0,
        market_premium=0.05,
        size_premium=0.02,
        value_premium=0.02,
        profitability_premium=0.02,
        investment_premium=0.02
    )
    assert abs(five_factor['expected_return'] - 0.03) < 1e-10  # Should equal risk-free rate
    assert five_factor['risk_assessment'] == "Low risk"


def test_vectorized_three_factor():
    """Test vectorized operations in three-factor model."""
    # Test with array inputs
    betas = np.array([
        [1.2, 0.5, 0.3],  # High risk portfolio
        [0.8, 0.4, 0.2],  # Moderate risk portfolio
        [0.5, 0.1, 0.1]   # Low risk portfolio
    ])
    
    result = fama_french_three_factor(
        risk_free_rate=0.03,
        market_beta=betas[:, 0],
        size_beta=betas[:, 1],
        value_beta=betas[:, 2],
        market_premium=0.05,
        size_premium=0.02,
        value_premium=0.03
    )
    
    # Check array outputs
    assert isinstance(result['expected_return'], np.ndarray)
    assert len(result['expected_return']) == 3
    
    # Check calculations for each portfolio
    expected_returns = np.array([0.109, 0.084, 0.060])  # Manually calculated
    np.testing.assert_array_almost_equal(result['expected_return'], expected_returns)
    
    # Check risk assessments
    assert isinstance(result['risk_assessment'], np.ndarray)
    assert result['risk_assessment'][0] == "Above-average risk"  # High risk portfolio
    assert result['risk_assessment'][1] == "Moderate risk"      # Moderate risk portfolio
    assert result['risk_assessment'][2] == "Low risk"          # Low risk portfolio


def test_vectorized_five_factor():
    """Test vectorized operations in five-factor model."""
    # Test with array inputs
    betas = np.array([
        [1.2, 0.5, 0.3, 0.2, 0.1],  # High risk portfolio
        [0.8, 0.4, 0.2, 0.2, 0.1],  # Moderate risk portfolio
        [0.5, 0.1, 0.1, 0.1, 0.1]   # Low risk portfolio
    ])
    
    result = fama_french_five_factor(
        risk_free_rate=0.03,
        market_beta=betas[:, 0],
        size_beta=betas[:, 1],
        value_beta=betas[:, 2],
        profitability_beta=betas[:, 3],
        investment_beta=betas[:, 4],
        market_premium=0.05,
        size_premium=0.02,
        value_premium=0.03,
        profitability_premium=0.01,
        investment_premium=0.01
    )
    
    # Check array outputs
    assert isinstance(result['expected_return'], np.ndarray)
    assert len(result['expected_return']) == 3
    
    # Check calculations for each portfolio
    expected_returns = np.array([0.112, 0.087, 0.062])  # Manually calculated
    np.testing.assert_array_almost_equal(result['expected_return'], expected_returns)
    
    # Check risk assessments
    assert isinstance(result['risk_assessment'], np.ndarray)
    assert result['risk_assessment'][0] == "Above-average risk"  # High risk portfolio
    assert result['risk_assessment'][1] == "Moderate risk"      # Moderate risk portfolio
    assert result['risk_assessment'][2] == "Low risk"          # Low risk portfolio


def test_broadcasting():
    """Test broadcasting capabilities."""
    # Test broadcasting with scalar and array inputs
    risk_free_rates = np.array([0.02, 0.03, 0.04])
    market_betas = np.array([1.0, 1.2, 0.8])
    
    # Three-factor model
    three_factor = fama_french_three_factor(
        risk_free_rate=risk_free_rates,
        market_beta=market_betas,
        size_beta=0.5,          # Scalar
        value_beta=0.3,         # Scalar
        market_premium=0.05,    # Scalar
        size_premium=0.02,      # Scalar
        value_premium=0.03      # Scalar
    )
    
    assert len(three_factor['expected_return']) == 3
    
    # Five-factor model
    five_factor = fama_french_five_factor(
        risk_free_rate=risk_free_rates,
        market_beta=market_betas,
        size_beta=0.5,              # Scalar
        value_beta=0.3,             # Scalar
        profitability_beta=0.2,     # Scalar
        investment_beta=0.1,        # Scalar
        market_premium=0.05,        # Scalar
        size_premium=0.02,          # Scalar
        value_premium=0.03,         # Scalar
        profitability_premium=0.01, # Scalar
        investment_premium=0.01     # Scalar
    )
    
    assert len(five_factor['expected_return']) == 3 