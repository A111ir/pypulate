"""
Logistic regression scoring for credit risk assessment.
"""

import numpy as np
from typing import Dict, Union, Any
from numpy.typing import ArrayLike


def logistic_regression_score(coefficients: ArrayLike, features: ArrayLike, 
                             intercept: float=0) -> Dict[str, Any]:
    """
    Calculate credit score using logistic regression coefficients.
    
    Parameters
    ----------
    coefficients : array_like
        Coefficients for each feature
    features : array_like
        Feature values
    intercept : float, optional
        Intercept term
        
    Returns
    -------
    dict
        Probability of default and score
    """
    coefficients = np.array(coefficients)
    features = np.array(features)
    
    log_odds = np.dot(coefficients, features) + intercept
    
    probability = 1 / (1 + np.exp(-log_odds))
    
    score = 850 - int(550 * probability)
    score = max(300, min(850, score))
    
    if score >= 750:
        risk_category = "Excellent"
    elif score >= 700:
        risk_category = "Good"
    elif score >= 650:
        risk_category = "Fair"
    elif score >= 600:
        risk_category = "Poor"
    else:
        risk_category = "Very Poor"
    
    result = {
        "probability_of_default": probability,
        "credit_score": score,
        "risk_category": risk_category,
        "log_odds": log_odds
    }
    
    return result 