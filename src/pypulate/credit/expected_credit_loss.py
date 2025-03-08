"""
Expected Credit Loss (ECL) calculation.
"""

import numpy as np
from typing import Dict, Union, Any


def expected_credit_loss(pd: float, lgd: float, ead: float, 
                        time_horizon: float=1.0, 
                        discount_rate: float=0.0) -> Dict[str, Any]:
    """
    Calculate expected credit loss.
    
    ECL = PD × LGD × EAD × Discount Factor
    
    Parameters
    ----------
    pd : float
        Probability of default
    lgd : float
        Loss given default (as a decimal)
    ead : float
        Exposure at default
    time_horizon : float, optional
        Time horizon in years
    discount_rate : float, optional
        Discount rate for future losses
        
    Returns
    -------
    dict
        ECL and components
    """
    if not 0 <= pd <= 1:
        raise ValueError("Probability of default must be between 0 and 1")
    if not 0 <= lgd <= 1:
        raise ValueError("Loss given default must be between 0 and 1")
    if ead < 0:
        raise ValueError("Exposure at default must be non-negative")
    
    discount_factor = 1 / (1 + discount_rate) ** time_horizon
    
    ecl = pd * lgd * ead * discount_factor
    
    expected_loss_rate = pd * lgd
    
    if expected_loss_rate < 0.01:
        risk_level = "Very Low"
    elif expected_loss_rate < 0.03:
        risk_level = "Low"
    elif expected_loss_rate < 0.07:
        risk_level = "Moderate"
    elif expected_loss_rate < 0.15:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    marginal_pd = 1 - (1 - pd) ** time_horizon
    lifetime_ecl = marginal_pd * lgd * ead * discount_factor
    
    result = {
        "expected_credit_loss": ecl,
        "lifetime_ecl": lifetime_ecl,
        "expected_loss_rate": expected_loss_rate,
        "risk_level": risk_level,
        "components": {
            "probability_of_default": pd,
            "loss_given_default": lgd,
            "exposure_at_default": ead,
            "discount_factor": discount_factor,
            "time_horizon": time_horizon
        }
    }
    
    return result 