"""
Risk-based loan pricing model.
"""

import numpy as np
from typing import Dict, Union, Any


def loan_pricing(loan_amount: float, term: float, pd: float, lgd: float, 
                funding_cost: float, operating_cost: float, 
                capital_requirement: float, target_roe: float) -> Dict[str, Any]:
    """
    Calculate risk-based loan pricing.
    
    Parameters
    ----------
    loan_amount : float
        Loan amount
    term : float
        Loan term in years
    pd : float
        Probability of default (annual)
    lgd : float
        Loss given default (as a decimal)
    funding_cost : float
        Cost of funds (annual rate)
    operating_cost : float
        Operating costs (as percentage of loan amount)
    capital_requirement : float
        Capital requirement as percentage of loan amount
    target_roe : float
        Target return on equity (annual rate)
        
    Returns
    -------
    dict
        Recommended interest rate and components
    """
    if not 0 <= pd <= 1:
        raise ValueError("Probability of default must be between 0 and 1")
    if not 0 <= lgd <= 1:
        raise ValueError("Loss given default must be between 0 and 1")
    if loan_amount <= 0 or term <= 0:
        raise ValueError("Loan amount and term must be positive")
    
    expected_loss_rate = pd * lgd
    expected_loss_component = expected_loss_rate / term
    
    funding_component = funding_cost
    
    operating_component = operating_cost / term
    
    capital_cost = capital_requirement * target_roe
    capital_component = capital_cost
    
    risk_premium = expected_loss_component + capital_component
    
    recommended_rate = funding_component + operating_component + risk_premium
    
    effective_annual_rate = (1 + recommended_rate / 12) ** 12 - 1
    
    monthly_rate = recommended_rate / 12
    num_payments = term * 12
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
    
    total_payments = monthly_payment * num_payments
    total_interest = total_payments - loan_amount
    
    expected_profit = total_interest - (loan_amount * expected_loss_rate) - (operating_cost * loan_amount * term)
    roi = expected_profit / (capital_requirement * loan_amount)
    
    result = {
        "recommended_rate": recommended_rate,
        "effective_annual_rate": effective_annual_rate,
        "monthly_payment": monthly_payment,
        "total_interest": total_interest,
        "expected_profit": expected_profit,
        "return_on_investment": roi,
        "components": {
            "expected_loss": expected_loss_component,
            "funding_cost": funding_component,
            "operating_cost": operating_component,
            "capital_cost": capital_component,
            "risk_premium": risk_premium
        }
    }
    
    return result 