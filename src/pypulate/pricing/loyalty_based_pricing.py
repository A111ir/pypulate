from typing import Dict, Tuple, Any

def calculate_loyalty_price(
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
        
    Returns
    -------
    dict
        Dictionary containing:
        - loyalty_price: final price after discount
        - loyalty_tier: the applicable tier
        - loyalty_discount: discount amount
        - additional_benefits: benefits dictionary
        
    """
    
    applicable_tier = 0
    applicable_discount = 0.0
    
    for tier, discount in sorted(loyalty_tiers.items()):
        if customer_tenure >= tier:
            applicable_tier = tier
            applicable_discount = discount
    
    loyalty_discount = base_price * applicable_discount
    
    loyalty_price = base_price - loyalty_discount

    return {
        'loyalty_price': loyalty_price,
        'loyalty_tier': applicable_tier,
        'loyalty_discount': loyalty_discount,
        'additional_benefits': additional_benefits
    }
