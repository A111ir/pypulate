import numpy as np

def calculate_time_based_price(
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

    Returns
    -------
    float
        Calculated price

    Examples
    --------
    >>> calculate_time_based_price(100, 2.5, 'hour')
    250.0  # 2.5 hours at $100/hour
    """
    if duration < minimum_duration:
        return 0
    

    if time_unit == 'minute':
        price = base_price * duration
    elif time_unit == 'hour':
        price = base_price * duration
    elif time_unit == 'day':
        price = base_price * duration
    else:
        raise ValueError(f"Unsupported time unit: {time_unit}")
    
    if rounding_method == 'up':
        return np.ceil(price)
    elif rounding_method == 'down':
        return np.floor(price)
    elif rounding_method == 'nearest':
        return np.round(price)
    else:
        raise ValueError(f"Unsupported rounding method: {rounding_method}")
    
    return price
