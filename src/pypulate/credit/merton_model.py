"""
Merton model for default probability calculation.
"""

import numpy as np
from scipy import stats
from typing import Dict, Union


def merton_model(asset_value: float, debt_face_value: float, 
                asset_volatility: float, risk_free_rate: float, 
                time_to_maturity: float) -> Dict[str, float]:
    """
    Calculate default probability using the Merton model.
    
    Parameters
    ----------
    asset_value : float
        Market value of assets
    debt_face_value : float
        Face value of debt
    asset_volatility : float
        Volatility of assets (annualized)
    risk_free_rate : float
        Risk-free interest rate
    time_to_maturity : float
        Time to maturity in years
        
    Returns
    -------
    dict
        Default probability and distance to default
    """
    d1 = (np.log(asset_value / debt_face_value) + 
          (risk_free_rate + 0.5 * asset_volatility**2) * time_to_maturity) / \
         (asset_volatility * np.sqrt(time_to_maturity))
    
    d2 = d1 - asset_volatility * np.sqrt(time_to_maturity)
    
    dd = d2
    
    pd = stats.norm.cdf(-d2)
    
    result = {
        "probability_of_default": pd,
        "distance_to_default": dd,
        "d1": d1,
        "d2": d2
    }
    
    return result 