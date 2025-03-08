"""
Loss Given Default (LGD) estimation.
"""

import numpy as np
from typing import Dict, Union, Any, Optional


def loss_given_default(collateral_value: float, loan_amount: float, 
                      recovery_rate: Optional[float]=None, 
                      liquidation_costs: float=0.1, 
                      time_to_recovery: float=1.0) -> Dict[str, Any]:
    """
    Estimate the loss given default for a loan.
    
    Parameters
    ----------
    collateral_value : float
        Value of collateral
    loan_amount : float
        Outstanding loan amount
    recovery_rate : float, optional
        Historical recovery rate for similar loans
    liquidation_costs : float, optional
        Costs associated with liquidating collateral
    time_to_recovery : float, optional
        Expected time to recovery in years
        
    Returns
    -------
    dict
        LGD estimate and components
    """
    if loan_amount <= 0:
        raise ValueError("Loan amount must be positive")
    if collateral_value < 0:
        raise ValueError("Collateral value cannot be negative")
    if recovery_rate is not None and not 0 <= recovery_rate <= 1:
        raise ValueError("Recovery rate must be between 0 and 1")
    if not 0 <= liquidation_costs <= 1:
        raise ValueError("Liquidation costs must be between 0 and 1")
    if time_to_recovery <= 0:
        raise ValueError("Time to recovery must be positive")
    
    net_collateral_value = collateral_value * (1 - liquidation_costs)
    
    ltv = loan_amount / collateral_value if collateral_value > 0 else float('inf')
    
    if net_collateral_value >= loan_amount:
        collateral_lgd = 0.0
    else:
        collateral_lgd = 1.0 - (net_collateral_value / loan_amount)
    
    if recovery_rate is not None:
        if ltv <= 0.5:
            weight_collateral = 0.8  
        elif ltv <= 0.8:
            weight_collateral = 0.6  
        else:
            weight_collateral = 0.4  
            
        weight_historical = 1.0 - weight_collateral
        
        lgd = (collateral_lgd * weight_collateral) + ((1 - recovery_rate) * weight_historical)
    else:
        lgd = collateral_lgd
    
    discount_rate = 0.05  
    time_value_factor = 1 / ((1 + discount_rate) ** time_to_recovery)
    
    present_value_loss = lgd * time_value_factor
    
    if lgd < 0.1:
        risk_level = "Very Low"
    elif lgd < 0.3:
        risk_level = "Low"
    elif lgd < 0.5:
        risk_level = "Moderate"
    elif lgd < 0.7:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    result = {
        "lgd": lgd,
        "present_value_lgd": present_value_loss,
        "risk_level": risk_level,
        "components": {
            "collateral_value": collateral_value,
            "net_collateral_value": net_collateral_value,
            "loan_amount": loan_amount,
            "loan_to_value": ltv,
            "collateral_lgd": collateral_lgd,
            "time_value_factor": time_value_factor
        }
    }
    
    if recovery_rate is not None:
        result["components"]["recovery_rate"] = recovery_rate
        result["components"]["weight_collateral"] = weight_collateral
        result["components"]["weight_historical"] = weight_historical
    
    return result 