from typing import List, Dict

def calculate_bundle_price(
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

    Returns
    -------
    float
        Total price for the bundle
        
    Examples
    --------
    >>> items = ['item1', 'item2', 'item3']
    >>> item_prices = {'item1': 10, 'item2': 20, 'item3': 30}
    >>> bundle_discounts = {'item1+item2': 0.10, 'item1+item2+item3': 0.20}
    >>> calculate_bundle_price(items, item_prices, bundle_discounts)
    48.0
    """
    total_price: float = 0.0
    bundle_counts: Dict[str, int] = {}

    for item in items:
        if item in item_prices:
            bundle_counts[item] = bundle_counts.get(item, 0) + 1
    
    for item, count in bundle_counts.items():
        total_price += item_prices[item] * count
    
    if len(items) >= minimum_bundle_size:
        max_discount = 0.0
        
        for bundle, discount in bundle_discounts.items():
            bundle_items = bundle.split('+')
            if all(item in bundle_counts for item in bundle_items):
                if discount > max_discount:
                    max_discount = discount
        
        if max_discount > 0:
            discount_amount = total_price * max_discount
            total_price -= discount_amount
    
    return total_price
