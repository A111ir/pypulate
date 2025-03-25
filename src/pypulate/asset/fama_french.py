"""
Fama-French factor models implementation.
"""

from typing import Dict, Any, Union, List
import numpy as np
from numpy.typing import ArrayLike


def fama_french_three_factor(
    risk_free_rate: Union[float, ArrayLike],
    market_beta: Union[float, ArrayLike],
    size_beta: Union[float, ArrayLike],
    value_beta: Union[float, ArrayLike],
    market_premium: Union[float, ArrayLike],
    size_premium: Union[float, ArrayLike],
    value_premium: Union[float, ArrayLike]
) -> Dict[str, Any]:
    """
    Calculate expected return using the Fama-French Three-Factor model.
    
    Formula: E(R) = Rf + β_m(Rm-Rf) + β_s(SMB) + β_v(HML)
    
    Parameters
    ----------
    risk_free_rate : float or array-like
        Risk-free rate of return (e.g., 0.03 for 3%)
    market_beta : float or array-like
        Beta coefficient for market risk factor
    size_beta : float or array-like
        Beta coefficient for size factor (SMB - Small Minus Big)
    value_beta : float or array-like
        Beta coefficient for value factor (HML - High Minus Low)
    market_premium : float or array-like
        Market risk premium (Rm - Rf)
    size_premium : float or array-like
        Size premium (SMB)
    value_premium : float or array-like
        Value premium (HML)
        
    Returns
    -------
    dict
        Expected return and factor contributions
        
    Raises
    ------
    ValueError
        If any of the input arrays are not of the same shape
    """
    inputs = [risk_free_rate, market_beta, size_beta, value_beta,
             market_premium, size_premium, value_premium]
    rf, b_m, b_s, b_v, p_m, p_s, p_v = [np.asarray(x) for x in inputs]
    
    market_contribution = b_m * p_m
    size_contribution = b_s * p_s
    value_contribution = b_v * p_v
    
    expected_return = rf + market_contribution + size_contribution + value_contribution
    
    total_systematic_risk = (np.abs(market_contribution) + 
                           np.abs(size_contribution) + 
                           np.abs(value_contribution))
    
    risk_assessment = np.where(
        total_systematic_risk <= 0.035, "Low risk",
        np.where(
            total_systematic_risk <= 0.07, "Moderate risk",
            np.where(
                total_systematic_risk <= 0.105, "Above-average risk",
                "High risk"
            )
        )
    )
    
    total_contribution = (np.abs(market_contribution) + 
                         np.abs(size_contribution) + 
                         np.abs(value_contribution))
    
    total_contribution = np.where(total_contribution > 0, total_contribution, 1.0)
    
    market_contribution_pct = np.abs(market_contribution) / total_contribution
    size_contribution_pct = np.abs(size_contribution) / total_contribution
    value_contribution_pct = np.abs(value_contribution) / total_contribution
    
    result = {
        "expected_return": expected_return,
        "risk_free_rate": rf,
        "total_systematic_risk": total_systematic_risk,
        "risk_assessment": risk_assessment,
        "factor_contributions": {
            "market": {
                "beta": b_m,
                "premium": p_m,
                "contribution": market_contribution,
                "contribution_pct": market_contribution_pct
            },
            "size": {
                "beta": b_s,
                "premium": p_s,
                "contribution": size_contribution,
                "contribution_pct": size_contribution_pct
            },
            "value": {
                "beta": b_v,
                "premium": p_v,
                "contribution": value_contribution,
                "contribution_pct": value_contribution_pct
            }
        }
    }
    
    return result


def fama_french_five_factor(
    risk_free_rate: Union[float, ArrayLike],
    market_beta: Union[float, ArrayLike],
    size_beta: Union[float, ArrayLike],
    value_beta: Union[float, ArrayLike],
    profitability_beta: Union[float, ArrayLike],
    investment_beta: Union[float, ArrayLike],
    market_premium: Union[float, ArrayLike],
    size_premium: Union[float, ArrayLike],
    value_premium: Union[float, ArrayLike],
    profitability_premium: Union[float, ArrayLike],
    investment_premium: Union[float, ArrayLike]
) -> Dict[str, Any]:
    """
    Calculate expected return using the Fama-French Five-Factor model.
    
    Formula: E(R) = Rf + β_m(Rm-Rf) + β_s(SMB) + β_v(HML) + β_p(RMW) + β_i(CMA)
    
    Parameters
    ----------
    risk_free_rate : float or array-like
        Risk-free rate of return (e.g., 0.03 for 3%)
    market_beta : float or array-like
        Beta coefficient for market risk factor
    size_beta : float or array-like
        Beta coefficient for size factor (SMB - Small Minus Big)
    value_beta : float or array-like
        Beta coefficient for value factor (HML - High Minus Low)
    profitability_beta : float or array-like
        Beta coefficient for profitability factor (RMW - Robust Minus Weak)
    investment_beta : float or array-like
        Beta coefficient for investment factor (CMA - Conservative Minus Aggressive)
    market_premium : float or array-like
        Market risk premium (Rm - Rf)
    size_premium : float or array-like
        Size premium (SMB)
    value_premium : float or array-like
        Value premium (HML)
    profitability_premium : float or array-like
        Profitability premium (RMW)
    investment_premium : float or array-like
        Investment premium (CMA)
        
    Returns
    -------
    dict
        Expected return and factor contributions
        
    Raises
    ------
    ValueError
        If any of the input arrays are not of the same shape
    """
    inputs = [risk_free_rate, market_beta, size_beta, value_beta,
             profitability_beta, investment_beta, market_premium,
             size_premium, value_premium, profitability_premium,
             investment_premium]
    (rf, b_m, b_s, b_v, b_p, b_i, p_m, p_s, p_v, p_p, p_i) = [
        np.asarray(x) for x in inputs
    ]
    
    market_contribution = b_m * p_m
    size_contribution = b_s * p_s
    value_contribution = b_v * p_v
    profitability_contribution = b_p * p_p
    investment_contribution = b_i * p_i
    
    expected_return = (rf + market_contribution + size_contribution +
                      value_contribution + profitability_contribution +
                      investment_contribution)
    
    total_systematic_risk = (np.abs(market_contribution) +
                           np.abs(size_contribution) +
                           np.abs(value_contribution) +
                           np.abs(profitability_contribution) +
                           np.abs(investment_contribution))
    
    risk_assessment = np.where(
        total_systematic_risk <= 0.05, "Low risk",
        np.where(
            total_systematic_risk <= 0.08, "Moderate risk",
            np.where(
                total_systematic_risk <= 0.12, "Above-average risk",
                "High risk"
            )
        )
    )
    
    total_contribution = (np.abs(market_contribution) +
                         np.abs(size_contribution) +
                         np.abs(value_contribution) +
                         np.abs(profitability_contribution) +
                         np.abs(investment_contribution))
    
    total_contribution = np.where(total_contribution > 0, total_contribution, 1.0)
    
    market_contribution_pct = np.abs(market_contribution) / total_contribution
    size_contribution_pct = np.abs(size_contribution) / total_contribution
    value_contribution_pct = np.abs(value_contribution) / total_contribution
    profitability_contribution_pct = np.abs(profitability_contribution) / total_contribution
    investment_contribution_pct = np.abs(investment_contribution) / total_contribution
    
    result = {
        "expected_return": expected_return,
        "risk_free_rate": rf,
        "total_systematic_risk": total_systematic_risk,
        "risk_assessment": risk_assessment,
        "factor_contributions": {
            "market": {
                "beta": b_m,
                "premium": p_m,
                "contribution": market_contribution,
                "contribution_pct": market_contribution_pct
            },
            "size": {
                "beta": b_s,
                "premium": p_s,
                "contribution": size_contribution,
                "contribution_pct": size_contribution_pct
            },
            "value": {
                "beta": b_v,
                "premium": p_v,
                "contribution": value_contribution,
                "contribution_pct": value_contribution_pct
            },
            "profitability": {
                "beta": b_p,
                "premium": p_p,
                "contribution": profitability_contribution,
                "contribution_pct": profitability_contribution_pct
            },
            "investment": {
                "beta": b_i,
                "premium": p_i,
                "contribution": investment_contribution,
                "contribution_pct": investment_contribution_pct
            }
        }
    }
    
    return result 