"""
Exposure at Default (EAD) calculation.
"""

import numpy as np
from typing import Dict, Union, Any


def exposure_at_default(current_balance: float, undrawn_amount: float, 
                       credit_conversion_factor: float=0.5) -> Dict[str, Any]:
    """
    Calculate exposure at default for credit facilities.
    
    Parameters
    ----------
    current_balance : float
        Current drawn balance
    undrawn_amount : float
        Undrawn commitment
    credit_conversion_factor : float, optional
        Factor to convert undrawn amounts to exposure
        
    Returns
    -------
    dict
        EAD and components
    """
    if current_balance < 0:
        raise ValueError("Current balance cannot be negative")
    if undrawn_amount < 0:
        raise ValueError("Undrawn amount cannot be negative")
    if not 0 <= credit_conversion_factor <= 1:
        raise ValueError("Credit conversion factor must be between 0 and 1")
    
    total_facility = current_balance + undrawn_amount
    
    utilization_rate = current_balance / total_facility if total_facility > 0 else 0
    
    ead = current_balance + (undrawn_amount * credit_conversion_factor)
    
    ead_percentage = ead / total_facility if total_facility > 0 else 0
    
    if utilization_rate < 0.3:
        regulatory_ccf = 0.2
    elif utilization_rate < 0.5:
        regulatory_ccf = 0.4
    elif utilization_rate < 0.7:
        regulatory_ccf = 0.6
    elif utilization_rate < 0.9:
        regulatory_ccf = 0.8
    else:
        regulatory_ccf = 1.0
    
    regulatory_ead = current_balance + (undrawn_amount * regulatory_ccf)
    
    stress_ccf = min(1.0, credit_conversion_factor * 1.5)
    stressed_ead = current_balance + (undrawn_amount * stress_ccf)
    
    if utilization_rate < 0.3:
        risk_level = "Low"
    elif utilization_rate < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    result = {
        "ead": ead,
        "regulatory_ead": regulatory_ead,
        "stressed_ead": stressed_ead,
        "ead_percentage": ead_percentage,
        "risk_level": risk_level,
        "components": {
            "current_balance": current_balance,
            "undrawn_amount": undrawn_amount,
            "total_facility": total_facility,
            "utilization_rate": utilization_rate,
            "credit_conversion_factor": credit_conversion_factor,
            "regulatory_ccf": regulatory_ccf,
            "stress_ccf": stress_ccf
        }
    }
    
    return result 