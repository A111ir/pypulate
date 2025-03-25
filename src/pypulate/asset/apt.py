"""
Arbitrage Pricing Theory (APT) model implementation.
"""

from typing import Dict, Any, List, Union
import numpy as np


def apt(
    risk_free_rate: float,
    factor_betas: List[float],
    factor_risk_premiums: List[float]
) -> Dict[str, Any]:
    """
    Calculate expected return using the Arbitrage Pricing Theory (APT) model.
    
    APT formula: E(R) = Rf + β₁(RP₁) + β₂(RP₂) + ... + βₙ(RPₙ)
    
    Parameters
    ----------
    risk_free_rate : float
        Risk-free rate of return (e.g., 0.03 for 3%)
    factor_betas : list of float
        Beta coefficients for each factor
    factor_risk_premiums : list of float
        Risk premiums for each factor
        
    Returns
    -------
    dict
        Expected return and components
        
    Raises
    ------
    ValueError
        If number of factor betas does not match number of factor risk premiums
        If no factors are provided
    -------
    
    """
    if len(factor_betas) != len(factor_risk_premiums):
        raise ValueError("Number of factor betas must match number of factor risk premiums")
    if len(factor_betas) == 0:
        raise ValueError("At least one factor must be provided")
    
    betas = np.array(factor_betas, dtype=np.float64)
    premiums = np.array(factor_risk_premiums, dtype=np.float64)
    
    factor_contributions = betas * premiums
    expected_return = risk_free_rate + np.sum(factor_contributions)
    
    abs_contributions = np.abs(factor_contributions)
    total_systematic_risk: float = np.sum(abs_contributions)
    
    if total_systematic_risk < 0.02:
        risk_assessment = "Low risk"
    elif total_systematic_risk < 0.05:
        risk_assessment = "Moderate risk"
    elif total_systematic_risk < 0.08:
        risk_assessment = "Above-average risk"
    else:
        risk_assessment = "High risk"
    
    total_contribution: float = np.sum(abs_contributions)
    factor_contribution_pct = abs_contributions / total_contribution if total_contribution > 0 else np.zeros_like(abs_contributions)
    
    factor_details = [
        {
            "factor_number": i + 1,
            "beta": float(beta),
            "risk_premium": float(premium),
            "contribution": float(contrib),
            "contribution_pct": float(pct)
        }
        for i, (beta, premium, contrib, pct) in enumerate(
            zip(betas, premiums, factor_contributions, factor_contribution_pct)
        )
    ]
    
    result = {
        "expected_return": float(expected_return),
        "risk_free_rate": risk_free_rate,
        "total_systematic_risk": float(total_systematic_risk),
        "risk_assessment": risk_assessment,
        "factor_details": factor_details
    }
    
    return result 