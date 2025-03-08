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

    Examples
    --------
    >>> base_features = ['feature1', 'feature2']
    >>> premium_features = ['feature3', 'feature4']
    >>> feature_usage = {'feature1': 10, 'feature2': 5, 'feature3': 3, 'feature4': 2}
    >>> free_limits = {'feature1': 10, 'feature2': 5}
    >>> overage_rates = {'feature1': 1.0, 'feature2': 0.5, 'feature3': 2.0, 'feature4': 1.5}
    >>> calculate_freemium_price(base_features, premium_features, feature_usage, free_limits, overage_rates)
    9.0
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
