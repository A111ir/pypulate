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
    # Convert usage time to minutes for easier comparison
    usage_hour, usage_minute = map(int, usage_time.split(':'))
    usage_time_in_minutes = usage_hour * 60 + usage_minute
    
    is_peak = False
    for weekday, (start, end) in peak_hours.items():
        # Convert start time to minutes
        start_hour, start_minute = map(int, start.split(':'))
        start_time_in_minutes = start_hour * 60 + start_minute
        
        # Convert end time to minutes
        end_hour, end_minute = map(int, end.split(':'))
        end_time_in_minutes = end_hour * 60 + end_minute
        
        # Check if usage time falls within this peak window
        if start_time_in_minutes <= usage_time_in_minutes < end_time_in_minutes:
            is_peak = True
            break
    
    if is_peak:
        return base_price * peak_multiplier
    else:
        return base_price * off_peak_multiplier
