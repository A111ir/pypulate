"""
Tiered Pricing Module

This module provides functions for calculating tiered pricing structures.
"""

from typing import Dict, Union, List
import numpy as np

def calculate_tiered_price(
    usage_units: float,
    tiers: Dict[str, float],
    cumulative: bool = True
) -> float:
    """
    Calculate price based on tiered pricing structure.
    
    Parameters
    ----------
    usage_units : float
        The number of units consumed
    tiers : dict
        Dictionary of tier ranges and their prices
        Format: {"0-1000": 0.10, "1001-2000": 0.08, "2001+": 0.05}
    cumulative : bool, default True
        If True, price is calculated cumulatively across tiers
        If False, entire usage is priced at the tier it falls into
        
    Returns
    -------
    float
        Total price based on tiered pricing
        
    Examples
    --------
    >>> tiers = {"0-1000": 0.10, "1001-2000": 0.08, "2001+": 0.05}
    >>> calculate_tiered_price(1500, tiers)
    130.0  # (1000 * 0.10) + (500 * 0.08)
    >>> calculate_tiered_price(1500, tiers, cumulative=False)
    120.0  # 1500 * 0.08
    """
    total_price = 0.0
    remaining_units = float(usage_units)
    
    parsed_tiers = []
    for tier_range, price in tiers.items():
        if '+' in tier_range:
            lower = float(tier_range.replace('+', ''))
            upper = float('inf')  
        else:
            lower, upper = map(float, tier_range.split('-'))
        parsed_tiers.append((lower, upper, price))
    
    sorted_tiers = sorted(parsed_tiers, key=lambda x: x[0])
    
    if not cumulative:
        for lower, upper, price in sorted_tiers:
            if usage_units <= upper:
                return usage_units * price
        return usage_units * sorted_tiers[-1][2]
    
    for lower, upper, price in sorted_tiers:
        if upper == float('inf'):
            tier_units = remaining_units
        else:
            tier_units = min(remaining_units, upper - lower + 1)
        if tier_units > 0:
            total_price += tier_units * price
            remaining_units -= tier_units
        if remaining_units <= 0:
            break
            
    return total_price 