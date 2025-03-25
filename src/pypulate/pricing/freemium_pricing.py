from typing import List, Dict

def calculate_freemium_price(
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
        
    Returns
    -------
    float
        Calculated price
    """
    total_price = 0.0
    for feature in base_features:
        if feature in feature_usage and feature in free_limits and feature in overage_rates:
            usage = feature_usage[feature]
            limit = free_limits[feature]
            if usage > limit:
                overage = usage - limit
                total_price += overage * overage_rates[feature]
    
    for feature in premium_features:
        if feature in feature_usage and feature in overage_rates:
            usage = feature_usage[feature]
            rate = overage_rates[feature]
            
            total_price += usage * rate
    
    return total_price
