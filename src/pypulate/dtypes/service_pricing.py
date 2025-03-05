"""
Service Pricing Module

This module provides a unified interface for various pricing calculations by combining
different pricing models from the pricing package.
"""

from typing import List, Dict, Union, Optional, Callable, Any
import numpy as np
from datetime import datetime, date

from ..pricing import (
    calculate_tiered_price,
    calculate_subscription_price,
    calculate_usage_price,
    calculate_volume_discount,
    calculate_peak_pricing,
    calculate_time_based_price,
    calculate_bundle_price,
    calculate_freemium_price,
    calculate_loyalty_price,
    apply_dynamic_pricing,
    PricingRule
)

class ServicePricing:
    """
    A class for calculating service pricing using various pricing models.
    
    This class provides a unified interface for:
    - Tiered pricing calculations
    - Subscription-based pricing
    - Usage-based pricing
    - Dynamic pricing adjustments
    - Volume discounts
    - Custom pricing rules
    
    Examples
    --------
    >>> from pypulate import ServicePricing
    >>> pricing = ServicePricing()
    >>> # Calculate tiered pricing
    >>> price = pricing.calculate_tiered_price(usage_units=1500, tiers={
    ...     "0-1000": 0.10,
    ...     "1001-2000": 0.08,
    ...     "2001+": 0.05
    ... })
    >>> print(f"Total price: ${price:.2f}")
    Total price: $130.00
    """
    
    def __init__(self):
        """Initialize the ServicePricing class."""
        self._state = {
            'current_pricing': {},
            'pricing_history': [],
            'custom_rules': PricingRule()
        }
    
    def calculate_tiered_price(
        self,
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
        """
        result = calculate_tiered_price(usage_units, tiers, cumulative)
        self._state['current_pricing']['tiered'] = result
        return result
    
    def calculate_subscription_price(
        self,
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
        """
        result = calculate_subscription_price(
            base_price, features, feature_prices, 
            duration_months, discount_rate
        )
        self._state['current_pricing']['subscription'] = result
        return result
    
    def calculate_usage_price(
        self,
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
        """
        result = calculate_usage_price(
            usage_metrics, metric_rates, 
            minimum_charge, maximum_charge
        )
        self._state['current_pricing']['usage'] = result
        return result
    
    def calculate_volume_discount(
        self,
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
        """
        result = calculate_volume_discount(base_price, volume, discount_tiers)
        self._state['current_pricing']['volume_discount'] = result
        return result
    
    def calculate_peak_pricing(
        self,
        base_price: float,
        usage_time: str,
        peak_hours: Dict[str, tuple],
        peak_multiplier: float = 1.5,
        off_peak_multiplier: float = 0.8
    ) -> float:
        """
        Calculate price based on peak/off-peak hours.
        
        Parameters
        ----------
        base_price : float
            Base price per unit
        usage_time : str
            Time of usage (format: "HH:MM")
        peak_hours : dict
            Dictionary of weekdays and their peak hours
            Format: {"monday": ("09:00", "17:00")}
        peak_multiplier : float, default 1.5
            Price multiplier during peak hours
        off_peak_multiplier : float, default 0.8
            Price multiplier during off-peak hours
        """
        result = calculate_peak_pricing(base_price, usage_time, peak_hours, peak_multiplier, off_peak_multiplier)
        self._state['current_pricing']['peak'] = result
        return result
    
    def calculate_time_based_price(
        self,
        base_price: float,
        duration: float,
        time_unit: str = 'hour',
        minimum_duration: float = 1.0,
        rounding_method: str = 'up'
    ) -> float:
        """
        Calculate price based on time duration.
        
        Parameters
        ----------
        base_price : float
            Base price per time unit
        duration : float
            Duration of usage
        time_unit : str
            Unit of time ('minute', 'hour', 'day')
        minimum_duration : float
            Minimum billable duration
        rounding_method : str
            How to round partial units ('up', 'down', 'nearest')
        """
        result = calculate_time_based_price(base_price, duration, time_unit, minimum_duration, rounding_method)
        self._state['current_pricing']['time_based'] = result
        return result
    
    def calculate_bundle_price(
        self,
        items: List[str],
        item_prices: Dict[str, float],
        bundle_discounts: Dict[str, float],
        minimum_bundle_size: int = 2
    ) -> float:
        """
        Calculate price for bundled items with discounts.

        Parameters
        ----------
        items : list
            List of items in the bundle
        item_prices : dict
            Individual prices for each item
        bundle_discounts : dict
            Discount rates for different bundle combinations
        minimum_bundle_size : int, default 2
            Minimum items required for bundle pricing
        """
        result = calculate_bundle_price(items, item_prices, bundle_discounts, minimum_bundle_size)
        self._state['current_pricing']['bundle'] = result
        return result
    
    def calculate_freemium_price(
        self,
        base_features: List[str],
        premium_features: List[str],
        feature_usage: Dict[str, float],
        free_limits: Dict[str, float],
        overage_rates: Dict[str, float]
    ) -> float:
        """
        Calculate price for freemium model with usage limits.
        
        Parameters
        ----------
        base_features : list
            List of free features
        premium_features : list
            List of premium features
        feature_usage : dict
            Usage metrics for each feature
        free_limits : dict
            Usage limits for free tier
        overage_rates : dict
            Rates for usage beyond free limits
        """
        result = calculate_freemium_price(
            base_features, premium_features, 
            feature_usage, free_limits, overage_rates
        )
        self._state['current_pricing']['freemium'] = result
        return result
    
    def calculate_loyalty_price(
        self,
        base_price: float,
        customer_tenure: int,
        loyalty_tiers: Dict[int, float],
        additional_benefits: Dict[str, float] = {}
    ) -> Dict[str, Any]:
        """
        Calculate price with loyalty discounts and benefits.

        Parameters
        ----------
        base_price : float
            Base price before loyalty benefits
        customer_tenure : int
            Customer's tenure in months
        loyalty_tiers : dict
            Discount rates for different tenure levels
        additional_benefits : dict, optional
            Additional benefits for loyal customers
        """
        result = calculate_loyalty_price(
            base_price, customer_tenure, 
            loyalty_tiers, additional_benefits
        )
        self._state['current_pricing']['loyalty'] = result
        return result
    
    def apply_dynamic_pricing(
        self,
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
        """
        result = apply_dynamic_pricing(
            base_price, demand_factor, competition_factor,
            seasonality_factor, min_price, max_price
        )
        self._state['current_pricing']['dynamic'] = result
        return result
    
    def add_custom_pricing_rule(
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
        """
        self._state['custom_rules'].add_rule(
            rule_name, calculation_function, description
        )
    
    def apply_custom_pricing_rule(
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
        """
        result = self._state['custom_rules'].apply_rule(rule_name, *args, **kwargs)
        self._state['current_pricing']['custom'] = result
        return result
    
    def get_pricing_history(self) -> List[Dict[str, float]]:
        """
        Get the history of pricing calculations.
        
        Returns
        -------
        list
            List of pricing calculations with their results
        """
        return self._state['pricing_history']
    
    def save_current_pricing(self) -> None:
        """Save the current pricing state to history."""
        if self._state['current_pricing']:
            self._state['pricing_history'].append({
                'timestamp': datetime.now(),
                'pricing': self._state['current_pricing'].copy()
            })
            self._state['current_pricing'] = {} 

    def get_current_pricing(self) -> Dict[str, float]:
        """Get the current pricing state."""
        return self._state['current_pricing']

