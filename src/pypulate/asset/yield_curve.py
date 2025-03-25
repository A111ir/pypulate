"""
Yield curve construction and interpolation functions.
"""

from typing import Dict, Any, List, Union, Optional, Tuple
import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def construct_yield_curve(
    maturities: List[float],
    rates: List[float],
    interpolation_method: str = 'cubic',
    extrapolate: bool = False
) -> Dict[str, Any]:
    """
    Construct a yield curve from observed market rates.
    
    Parameters
    ----------
    maturities : list of float
        Maturities in years for the observed rates
    rates : list of float
        Observed interest rates (as decimals)
    interpolation_method : str, optional
        Method for interpolation ('linear', 'cubic', 'monotonic'), by default 'cubic'
    extrapolate : bool, optional
        Whether to allow extrapolation beyond observed maturities, by default False
        
    Returns
    -------
    dict
        Yield curve object and details
        
    Raises
    ------
    ValueError
        If inputs are invalid or maturities are not in ascending order
        If interpolation method is not valid
    """
    # Validate inputs
    if len(maturities) != len(rates):
        raise ValueError("Length of maturities and rates must be the same")
    if len(maturities) < 2:
        raise ValueError("At least two points are required to construct a yield curve")
    if not all(m > 0 for m in maturities):
        raise ValueError("All maturities must be positive")
    if not all(r >= 0 for r in rates):
        raise ValueError("All rates must be non-negative")
    if not all(maturities[i] < maturities[i+1] for i in range(len(maturities)-1)):
        raise ValueError("Maturities must be in ascending order")
    
    valid_methods = ['linear', 'cubic', 'monotonic']
    if interpolation_method not in valid_methods:
        raise ValueError(f"interpolation_method must be one of {valid_methods}")
    
    # Convert inputs to numpy arrays
    maturities_array = np.array(maturities)
    rates_array = np.array(rates)
    
    # Create interpolation function
    if interpolation_method == 'linear':
        interpolate_func = interp1d(maturities_array, rates_array, 
                                   kind='linear', bounds_error=not extrapolate, 
                                   fill_value=('extrapolate' if extrapolate else np.nan))
    elif interpolation_method == 'cubic':
        interpolate_func = CubicSpline(maturities_array, rates_array, 
                                      extrapolate=extrapolate)
    elif interpolation_method == 'monotonic':
        # Monotonic cubic interpolation (PCHIP)
        interpolate_func = interp1d(maturities_array, rates_array, 
                                   kind='cubic', bounds_error=not extrapolate, 
                                   fill_value=('extrapolate' if extrapolate else np.nan),
                                   assume_sorted=True)
    
    # Calculate curve characteristics
    min_maturity = min(maturities)
    max_maturity = max(maturities)
    
    # Calculate curve steepness (long-term rate minus short-term rate)
    steepness = rates[-1] - rates[0]
    
    # Calculate average rate
    average_rate = np.mean(rates)
    
    # Calculate forward rates between consecutive maturities using vectorized operations
    # Instead of looping through each pair of maturities
    t1 = maturities_array[:-1]  # Start maturities
    t2 = maturities_array[1:]   # End maturities
    r1 = rates_array[:-1]       # Rates at start maturities
    r2 = rates_array[1:]        # Rates at end maturities
    
    # Calculate implied forward rates using vectorized operations
    # Formula: forward_rate = ((1 + r2)^t2 / (1 + r1)^t1)^(1/(t2-t1)) - 1
    forward_rates_array = ((1 + r2) ** t2 / (1 + r1) ** t1) ** (1 / (t2 - t1)) - 1
    
    # Create the forward rates list of dictionaries
    forward_rates = [
        {
            'start_maturity': t1[i],
            'end_maturity': t2[i],
            'forward_rate': forward_rates_array[i]
        }
        for i in range(len(t1))
    ]
    
    # Return results
    result = {
        "maturities": maturities,
        "rates": rates,
        "interpolation_method": interpolation_method,
        "extrapolate": extrapolate,
        "interpolate_func": interpolate_func,
        "min_maturity": min_maturity,
        "max_maturity": max_maturity,
        "steepness": steepness,
        "average_rate": average_rate,
        "forward_rates": forward_rates
    }
    
    return result


def interpolate_rate(
    yield_curve: Dict[str, Any],
    maturity: float
) -> float:
    """
    Interpolate interest rate at a specific maturity from a yield curve.
    
    Parameters
    ----------
    yield_curve : dict
        Yield curve object from construct_yield_curve
    maturity : float
        Maturity in years for which to interpolate the rate
        
    Returns
    -------
    float
        Interpolated interest rate
        
    Raises
    ------
    ValueError
        If yield_curve is not a valid yield curve object
        If maturity is not positive
    """
    if not isinstance(yield_curve, dict) or 'interpolate_func' not in yield_curve:
        raise ValueError("yield_curve must be a valid yield curve object from construct_yield_curve")
    if maturity <= 0:
        raise ValueError("Maturity must be positive")
    
    if not yield_curve['extrapolate']:
        min_maturity = yield_curve['min_maturity']
        max_maturity = yield_curve['max_maturity']
        if maturity < min_maturity or maturity > max_maturity:
            raise ValueError(f"Maturity {maturity} is outside the range [{min_maturity}, {max_maturity}] "
                            f"and extrapolation is not allowed")
    
    interpolated_rate = float(yield_curve['interpolate_func'](maturity))
    
    return interpolated_rate 