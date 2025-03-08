from typing import Dict, Tuple, Optional

def calculate_peak_pricing(
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
        
    Returns
    -------
    float
        Calculated price
    
    Examples
    --------
    >>> calculate_peak_pricing(100, "10:00", {"monday": ("09:00", "17:00")})
    150.0  # $100 * 1.5
    """
    usage_hour, usage_minute = map(int, usage_time.split(':'))
    
    is_peak = False
    for weekday, (start, end) in peak_hours.items():
        start_hour, start_minute = map(int, start.split(':'))
        end_hour, end_minute = map(int, end.split(':'))
        
        if weekday == "monday" and start_hour <= usage_hour < end_hour:
            is_peak = True
            break
        elif weekday == "tuesday" and start_hour <= usage_hour < end_hour:
            is_peak = True
            break
        elif weekday == "wednesday" and start_hour <= usage_hour < end_hour:
            is_peak = True
            break
        elif weekday == "thursday" and start_hour <= usage_hour < end_hour:
            is_peak = True
            break
        elif weekday == "friday" and start_hour <= usage_hour < end_hour:
            is_peak = True
            break
        elif weekday == "saturday" and start_hour <= usage_hour < end_hour:
            is_peak = True
            break
        elif weekday == "sunday" and start_hour <= usage_hour < end_hour:
            is_peak = True
            break
    
    

    if is_peak:
        return base_price * peak_multiplier
    else:
        return base_price * off_peak_multiplier
