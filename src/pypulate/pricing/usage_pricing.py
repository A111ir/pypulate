"""
Usage Pricing Module

This module provides functions for calculating usage-based pricing.
"""

from typing import Dict, Optional, Union
import numpy as np

def calculate_usage_price(
    usage_metrics: Dict[str, float],
    metric_rates: Dict[str, float],
    minimum_charge: float = 0.0,
    maximum_charge: Optional[float] = None
) -> float:
    """
    Calculate price based on usage metrics.
    
    Parameters
    ----------
    usage_metrics : dict
        Dictionary of metric names and their usage values
    metric_rates : dict
        Dictionary of metric names and their per-unit rates
    minimum_charge : float, default 0.0
        Minimum charge to apply
    maximum_charge : float, optional
        Maximum charge cap
        
    Returns
    -------
    float
        Total usage-based price
        
    Examples
    --------
    >>> metrics = {'api_calls': 1000, 'storage_gb': 50}
    >>> rates = {'api_calls': 0.001, 'storage_gb': 0.10}
    >>> calculate_usage_price(metrics, rates)
    6.0  # (1000 * 0.001) + (50 * 0.10)
    >>> calculate_usage_price(metrics, rates, minimum_charge=10.0)
    10.0  # Max of calculated price and minimum charge
    """
    total_price = sum(
        usage * metric_rates.get(metric, 0)
        for metric, usage in usage_metrics.items()
    )
    
    total_price = max(total_price, minimum_charge)
    
    if maximum_charge is not None:
        total_price = min(total_price, maximum_charge)
        
    return total_price

def calculate_volume_discount(
    base_price: float,
    volume: int,
    discount_tiers: Dict[int, float]
) -> float:
    """
    Calculate price with volume-based discounts.
    
    Parameters
    ----------
    base_price : float
        Base price per unit
    volume : int
        Number of units
    discount_tiers : dict
        Dictionary of volume thresholds and discount rates
        Format: {100: 0.05, 500: 0.10, 1000: 0.15}
        
    Returns
    -------
    float
        Total price after volume discount
        
    Examples
    --------
    >>> tiers = {100: 0.05, 500: 0.10, 1000: 0.15}
    >>> calculate_volume_discount(10.0, 750, tiers)
    6750.0  # 750 * 10.0 * (1 - 0.10)
    """
    discount_rate = 0.0
    for threshold, rate in sorted(discount_tiers.items()):
        if volume >= threshold:
            discount_rate = rate
        else:
            break
            
    return base_price * volume * (1 - discount_rate) 