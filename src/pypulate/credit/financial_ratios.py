"""
Financial ratios calculation for credit assessment.
"""

from typing import Dict, Union, Any


def financial_ratios(current_assets: float, current_liabilities: float, 
                    total_assets: float, total_liabilities: float, 
                    ebit: float, interest_expense: float, 
                    net_income: float, total_equity: float, 
                    sales: float) -> Dict[str, Any]:
    """
    Calculate key financial ratios for credit assessment.
    
    Parameters
    ----------
    current_assets : float
        Current assets
    current_liabilities : float
        Current liabilities
    total_assets : float
        Total assets
    total_liabilities : float
        Total liabilities
    ebit : float
        Earnings before interest and taxes
    interest_expense : float
        Interest expense
    net_income : float
        Net income
    total_equity : float
        Total equity
    sales : float
        Sales
        
    Returns
    -------
    dict
        Financial ratios and assessments
    """
    current_ratio = current_assets / current_liabilities
    
    debt_ratio = total_liabilities / total_assets
    debt_to_equity = total_liabilities / total_equity
    
    return_on_assets = net_income / total_assets
    return_on_equity = net_income / total_equity
    
    interest_coverage = ebit / interest_expense
    
    asset_turnover = sales / total_assets
    
    liquidity_assessment = "Strong" if current_ratio >= 2 else "Adequate" if current_ratio >= 1 else "Weak"
    solvency_assessment = "Strong" if debt_ratio <= 0.4 else "Adequate" if debt_ratio <= 0.6 else "Weak"
    profitability_assessment = "Strong" if return_on_equity >= 0.15 else "Adequate" if return_on_equity >= 0.08 else "Weak"
    coverage_assessment = "Strong" if interest_coverage >= 3 else "Adequate" if interest_coverage >= 1.5 else "Weak"
    
    assessments = [liquidity_assessment, solvency_assessment, profitability_assessment, coverage_assessment]
    if assessments.count("Strong") >= 3:
        overall = "Strong financial position"
    elif assessments.count("Weak") >= 3:
        overall = "Weak financial position"
    else:
        overall = "Adequate financial position"
    
    result = {
        "liquidity": {
            "current_ratio": current_ratio,
            "assessment": liquidity_assessment
        },
        "solvency": {
            "debt_ratio": debt_ratio,
            "debt_to_equity": debt_to_equity,
            "assessment": solvency_assessment
        },
        "profitability": {
            "return_on_assets": return_on_assets,
            "return_on_equity": return_on_equity,
            "assessment": profitability_assessment
        },
        "coverage": {
            "interest_coverage": interest_coverage,
            "assessment": coverage_assessment
        },
        "efficiency": {
            "asset_turnover": asset_turnover
        },
        "overall_assessment": overall
    }
    
    return result 