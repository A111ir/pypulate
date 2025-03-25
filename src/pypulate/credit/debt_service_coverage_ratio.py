"""
Debt Service Coverage Ratio (DSCR) calculation.
"""
from typing import Union

def debt_service_coverage_ratio(net_operating_income, total_debt_service) -> dict[str, Union[float, str]]:
    """
    Calculate Debt Service Coverage Ratio (DSCR).
    
    DSCR = Net Operating Income / Total Debt Service
    
    Parameters
    ----------
    net_operating_income : float
        Net operating income
    total_debt_service : float
        Total debt service
        
    Returns
    -------
    dict
        DSCR value and interpretation
    """
    dscr = net_operating_income / total_debt_service
    
    if dscr < 1.0:
        assessment = "Negative cash flow, high risk"
        rating = "Poor"
    elif dscr < 1.25:
        assessment = "Barely sufficient, moderate risk"
        rating = "Fair"
    elif dscr < 1.5:
        assessment = "Sufficient coverage, acceptable risk"
        rating = "Good"
    else:
        assessment = "Strong coverage, low risk"
        rating = "Excellent"
        
    result = {
        "dscr": dscr,
        "assessment": assessment,
        "rating": rating
    }
    
    return result 