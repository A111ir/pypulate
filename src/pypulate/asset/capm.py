"""
Capital Asset Pricing Model (CAPM) implementation.
"""

from typing import Dict, Any, Union, List, Optional
import numpy as np
from numpy.typing import ArrayLike


def capm(
    risk_free_rate: Union[float, ArrayLike],
    beta: Union[float, ArrayLike],
    market_return: Union[float, ArrayLike]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Calculate expected return using the Capital Asset Pricing Model (CAPM).
    
    CAPM formula: E(R) = Rf + β × (Rm - Rf)
    
    This implementation supports both scalar and vector inputs for batch calculations.
    
    Parameters
    ----------
    risk_free_rate : float or array-like
        Risk-free rate of return (e.g., 0.03 for 3%)
    beta : float or array-like
        Beta of the asset(s) (measure of systematic risk)
    market_return : float or array-like
        Expected market return (e.g., 0.08 for 8%)
        
    Returns
    -------
    dict or list of dict
        Expected return and components for each asset
        
    Raises
    ------
    ValueError
        If any of the input parameters are invalid
    """
    rf = np.asarray(risk_free_rate, dtype=np.float64)
    β = np.asarray(beta, dtype=np.float64)
    rm = np.asarray(market_return, dtype=np.float64)
    
    if np.any(rf < 0):
        raise ValueError("Risk-free rate cannot be negative")
    
    shapes = [s.shape for s in [rf, β, rm] if not np.isscalar(s)]
    if shapes:
        output_shape = np.broadcast_shapes(*shapes)
        rf = np.broadcast_to(rf, output_shape)
        β = np.broadcast_to(β, output_shape)
        rm = np.broadcast_to(rm, output_shape)
    
    market_risk_premium = rm - rf
    expected_return = rf + β * market_risk_premium
    
    risk_assessment = np.select(
        condlist=[
            β < 0.8,
            β < 1.0,
            β < 1.2,
            β < 1.5
        ],
        choicelist=[
            "Low risk",
            "Below-market risk",
            "Market-level risk",
            "Above-market risk"
        ],
        default="High risk"
    )
    
    if all(np.isscalar(x) or x.size == 1 for x in [rf, β, rm]):
        return {
            "expected_return": float(expected_return.item()),
            "risk_free_rate": float(rf.item()),
            "beta": float(β.item()),
            "market_return": float(rm.item()),
            "market_risk_premium": float(market_risk_premium.item()),
            "risk_assessment": str(risk_assessment.item())
        }
    else:
        return [
            {
                "expected_return": float(er),
                "risk_free_rate": float(rf_i),
                "beta": float(β_i),
                "market_return": float(rm_i),
                "market_risk_premium": float(mrp),
                "risk_assessment": str(ra)
            }
            for er, rf_i, β_i, rm_i, mrp, ra in zip(
                expected_return.flatten(),
                rf.flatten(),
                β.flatten(),
                rm.flatten(),
                market_risk_premium.flatten(),
                risk_assessment.flatten()
            )
        ] 