"""
Dynamic Pricing Module

This module provides functions for calculating dynamic pricing adjustments.
"""

from typing import Optional, Union, Callable, Dict
import numpy as np

def apply_dynamic_pricing(
    base_price: float,
    demand_factor: float,
    competition_factor: float,
    seasonality_factor: float = 1.0,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> float:
    """
    Calculate dynamically adjusted price based on market factors.
    
    Parameters
    ----------
    base_price : float
        Base price before adjustments
    demand_factor : float
        Demand multiplier (1.0 is neutral)
    competition_factor : float
        Competition multiplier (1.0 is neutral)
    seasonality_factor : float, default 1.0
        Seasonal adjustment factor
    min_price : float, optional
        Minimum price floor
    max_price : float, optional
        Maximum price ceiling
        
    Returns
    -------
    float
        Dynamically adjusted price
        
    Examples
    --------
    >>> apply_dynamic_pricing(100.0, 1.2, 0.9, 1.1)
    118.8  # 100.0 * 1.2 * 0.9 * 1.1
    >>> apply_dynamic_pricing(100.0, 1.5, 0.8, min_price=90.0, max_price=150.0)
    120.0  # 100.0 * 1.5 * 0.8, bounded by min/max
    """
    adjusted_price = base_price * demand_factor * competition_factor * seasonality_factor
    
    if min_price is not None:
        adjusted_price = max(adjusted_price, min_price)
    if max_price is not None:
        adjusted_price = min(adjusted_price, max_price)
        
    return adjusted_price

class PricingRule:
    """
    A class for managing custom pricing rules.
    
    This class provides methods for:
    - Adding custom pricing rules
    - Applying custom pricing rules
    - Managing rule metadata
    """
    
    def __init__(self):
        """Initialize the PricingRule class."""
        self._rules: Dict[str, Dict] = {}
    
    def add_rule(
        self,
        rule_name: str,
        calculation_function: Callable[..., float],
        description: str = ""
    ) -> None:
        """
        Add a custom pricing rule.
        
        Parameters
        ----------
        rule_name : str
            Name of the custom pricing rule
        calculation_function : callable
            Function that implements the custom pricing logic
        description : str, optional
            Description of the pricing rule
            
        Examples
        --------
        >>> def holiday_pricing(base_price, holiday_multiplier):
        ...     return base_price * holiday_multiplier
        >>> rules = PricingRule()
        >>> rules.add_rule('holiday', holiday_pricing, 'Apply holiday pricing multiplier')
        """
        self._rules[rule_name] = {
            'function': calculation_function,
            'description': description
        }
    
    def apply_rule(
        self,
        rule_name: str,
        *args,
        **kwargs
    ) -> float:
        """
        Apply a custom pricing rule.
        
        Parameters
        ----------
        rule_name : str
            Name of the custom pricing rule
        *args, **kwargs
            Arguments to pass to the custom pricing function
            
        Returns
        -------
        float
            Price calculated using the custom rule
            
        Raises
        ------
        KeyError
            If the specified rule_name doesn't exist
            
        Examples
        --------
        >>> rules = PricingRule()
        >>> rules.add_rule('holiday', lambda price, mult: price * mult)
        >>> rules.apply_rule('holiday', 100.0, 1.2)
        120.0
        """
        if rule_name not in self._rules:
            raise KeyError(f"Custom pricing rule '{rule_name}' not found")
            
        return self._rules[rule_name]['function'](*args, **kwargs)
    
    def get_rule_description(self, rule_name: str) -> str:
        """
        Get the description of a pricing rule.
        
        Parameters
        ----------
        rule_name : str
            Name of the pricing rule
            
        Returns
        -------
        str
            Description of the pricing rule
            
        Raises
        ------
        KeyError
            If the specified rule_name doesn't exist
        """
        if rule_name not in self._rules:
            raise KeyError(f"Custom pricing rule '{rule_name}' not found")
            
        return self._rules[rule_name]['description']
    
    def list_rules(self) -> Dict[str, str]:
        """
        List all available pricing rules and their descriptions.
        
        Returns
        -------
        dict
            Dictionary of rule names and their descriptions
        """
        return {name: rule['description'] for name, rule in self._rules.items()} 