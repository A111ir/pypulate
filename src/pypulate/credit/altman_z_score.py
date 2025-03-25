"""
Altman Z-Score for bankruptcy prediction.
"""

from typing import Union


def altman_z_score(working_capital, retained_earnings, ebit, 
                  market_value_equity, sales, total_assets, total_liabilities) -> dict[str, Union[float, str]]:
    """
    Calculate Altman Z-Score for predicting bankruptcy risk.
    
    Z-Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
    
    Parameters
    ----------
    working_capital : float
        Working capital
    retained_earnings : float
        Retained earnings
    ebit : float
        Earnings before interest and taxes
    market_value_equity : float
        Market value of equity
    sales : float
        Sales
    total_assets : float
        Total assets
    total_liabilities : float
        Total liabilities
        
    Returns
    -------
    dict
        Z-score value and risk interpretation
    """
    x1 = working_capital / total_assets  
    x2 = retained_earnings / total_assets 
    x3 = ebit / total_assets 
    x4 = market_value_equity / total_liabilities  
    x5 = sales / total_assets  
    
    z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 0.999*x5
    
    if z_score < 1.81:
        risk = "High risk of bankruptcy"
        zone = "Distress"
    elif z_score < 2.99:
        risk = "Grey area, moderate risk"
        zone = "Grey"
    else:
        risk = "Low risk of bankruptcy"
        zone = "Safe"
        
    result = {
        "z_score": z_score,
        "risk_assessment": risk,
        "zone": zone,
        "components": {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5
        }
    }
    
    return result 