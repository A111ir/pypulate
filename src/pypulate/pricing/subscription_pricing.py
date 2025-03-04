"""
Subscription Pricing Module

This module provides functions for calculating subscription-based pricing.
"""

from typing import List, Dict, Union
import numpy as np

def calculate_subscription_price(
    base_price: float,
    features: List[str],
    feature_prices: Dict[str, float],
    duration_months: int = 1,
    discount_rate: float = 0.0
) -> float:
    """
    Calculate subscription price including selected features.
    
    Parameters
    ----------
    base_price : float
        Base subscription price
    features : list
        List of selected feature names
    feature_prices : dict
        Dictionary of feature names and their prices
    duration_months : int, default 1
        Subscription duration in months
    discount_rate : float, default 0.0
        Annual discount rate for longer subscriptions
        
    Returns
    -------
    float
        Total subscription price
        
    Examples
    --------
    >>> features = ['premium', 'api_access']
    >>> feature_prices = {'premium': 49.99, 'api_access': 29.99}
    >>> calculate_subscription_price(99.99, features, feature_prices)
    179.97  # 99.99 + 49.99 + 29.99
    >>> calculate_subscription_price(99.99, features, feature_prices, 
    ...                            duration_months=12, discount_rate=0.10)
    1943.68  # (99.99 + 49.99 + 29.99) * 12 * (1 - 0.10)
    """
    feature_cost = sum(feature_prices.get(feature, 0) for feature in features)
    
    monthly_price = base_price + feature_cost
    
    # Apply duration discount
    if duration_months > 1:
        annual_discount = (1 - discount_rate) ** (duration_months / 12)
        return monthly_price * duration_months * annual_discount
    
    return monthly_price * duration_months 