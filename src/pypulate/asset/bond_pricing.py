"""
Bond pricing and fixed income analysis functions.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from scipy.optimize import newton


def price_bond(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    yield_to_maturity: float,
    frequency: int = 2
) -> Dict[str, Any]:
    """
    Calculate the price of a bond using discounted cash flow analysis.
    
    Parameters
    ----------
    face_value : float
        Face value (par value) of the bond
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.05 for 5%)
    years_to_maturity : float
        Years until the bond matures
    yield_to_maturity : float
        Annual yield to maturity as a decimal (e.g., 0.06 for 6%)
    frequency : int, optional
        Number of coupon payments per year, by default 2 (semi-annual)
        
    Returns
    -------
    dict
        Bond price and details
        
    Raises
    ------
    ValueError
        If any of the input parameters are invalid
    """
    if face_value <= 0:
        raise ValueError("Face value must be positive")
    if coupon_rate < 0:
        raise ValueError("Coupon rate cannot be negative")
    if years_to_maturity <= 0:
        raise ValueError("Years to maturity must be positive")
    if yield_to_maturity < 0:
        raise ValueError("Yield to maturity cannot be negative")
    if frequency <= 0 or not isinstance(frequency, int):
        raise ValueError("Frequency must be a positive integer")
    
    periodic_coupon_rate = coupon_rate / frequency
    periodic_yield = yield_to_maturity / frequency
    periods = int(years_to_maturity * frequency)
    coupon_payment = face_value * periodic_coupon_rate
    
    if periods == 0:
        price = face_value
    else:
        t = np.arange(1, periods + 1)
        
        discount_factors = (1 + periodic_yield) ** (-t)
        
        cash_flows = np.full(periods, coupon_payment)
        cash_flows[-1] += face_value  
        
        price = np.sum(cash_flows * discount_factors)
    
    current_yield = (coupon_rate * face_value) / price if price > 0 else 0
    
    if abs(price - face_value) < 0.01:
        status = "At par"
    elif price > face_value:
        status = "Trading at premium"
    else:
        status = "Trading at discount"
    
    result = {
        "price": price,
        "face_value": face_value,
        "coupon_rate": coupon_rate,
        "years_to_maturity": years_to_maturity,
        "yield_to_maturity": yield_to_maturity,
        "frequency": frequency,
        "coupon_payment": coupon_payment,
        "current_yield": current_yield,
        "status": status
    }
    
    return result


def yield_to_maturity(
    price: float,
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    frequency: int = 2,
    initial_guess: float = 0.05,
    precision: float = 1e-10,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    Calculate the yield to maturity of a bond.
    
    Parameters
    ----------
    price : float
        Current market price of the bond
    face_value : float
        Face value (par value) of the bond
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.05 for 5%)
    years_to_maturity : float
        Years until the bond matures
    frequency : int, optional
        Number of coupon payments per year, by default 2 (semi-annual)
    initial_guess : float, optional
        Initial guess for YTM, by default 0.05 (5%)
    precision : float, optional
        Desired precision for YTM calculation, by default 1e-10
    max_iterations : int, optional
        Maximum number of iterations, by default 100
        
    Returns
    -------
    dict
        Yield to maturity and bond details
        
    Raises
    ------
    ValueError
        If any of the input parameters are invalid
    """
    if price <= 0 or face_value <= 0:
        raise ValueError("Price and face value must be positive")
    if coupon_rate < 0:
        raise ValueError("Coupon rate cannot be negative")
    if years_to_maturity <= 0:
        raise ValueError("Years to maturity must be positive")
    if frequency <= 0 or not isinstance(frequency, int):
        raise ValueError("Frequency must be a positive integer")
    
    def objective(ytm):
        bond_price = price_bond(
            face_value=face_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=ytm,
            frequency=frequency
        )['price']
        return bond_price - price
    
    try:
        ytm = newton(
            func=objective,
            x0=initial_guess,
            tol=precision,
            maxiter=max_iterations
        )
        
        bond_details = price_bond(
            face_value=face_value,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            yield_to_maturity=ytm,
            frequency=frequency
        )
        
        result = {
            "yield_to_maturity": ytm,
            "price": price,
            "face_value": face_value,
            "coupon_rate": coupon_rate,
            "years_to_maturity": years_to_maturity,
            "frequency": frequency,
            "coupon_payment": bond_details['coupon_payment'],
            "current_yield": bond_details['current_yield'],
            "status": bond_details['status']
        }
        
        return result
    
    except RuntimeError:
        return {
            "error": "Could not converge to a solution",
            "price": price,
            "face_value": face_value,
            "coupon_rate": coupon_rate,
            "years_to_maturity": years_to_maturity,
            "frequency": frequency
        }


def duration_convexity(
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    yield_to_maturity: float,
    frequency: int = 2
) -> Dict[str, Any]:
    """
    Calculate the Macaulay duration, modified duration, and convexity of a bond.
    
    Parameters
    ----------
    face_value : float
        Face value (par value) of the bond
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.05 for 5%)
    years_to_maturity : float
        Years until the bond matures
    yield_to_maturity : float
        Annual yield to maturity as a decimal (e.g., 0.06 for 6%)
    frequency : int, optional
        Number of coupon payments per year, by default 2 (semi-annual)
        
    Returns
    -------
    dict
        Duration, modified duration, convexity, and bond details
    """
    if face_value <= 0:
        raise ValueError("Face value must be positive")
    if coupon_rate < 0:
        raise ValueError("Coupon rate cannot be negative")
    if years_to_maturity <= 0:
        raise ValueError("Years to maturity must be positive")
    if yield_to_maturity < 0:
        raise ValueError("Yield to maturity cannot be negative")
    if frequency <= 0 or not isinstance(frequency, int):
        raise ValueError("Frequency must be a positive integer")
    
    bond_price_result = price_bond(
        face_value=face_value,
        coupon_rate=coupon_rate,
        years_to_maturity=years_to_maturity,
        yield_to_maturity=yield_to_maturity,
        frequency=frequency
    )
    price = bond_price_result['price']
    
    periodic_coupon_rate = coupon_rate / frequency
    periodic_yield = yield_to_maturity / frequency
    coupon_payment = face_value * periodic_coupon_rate
    periods = int(years_to_maturity * frequency)
    
    t = np.arange(1, periods + 1) / frequency
    
    cash_flows = np.full(periods, coupon_payment)
    cash_flows[-1] += face_value 
    
    discount_factors = (1 + periodic_yield) ** (-t * frequency)
    
    present_values = cash_flows * discount_factors
    
    macaulay_duration = np.sum(t * present_values) / price
    
    modified_duration = macaulay_duration / (1 + periodic_yield)
    
    t_squared = t ** 2
    t_plus_one = t * (t + 1/frequency)
    convexity = np.sum((t_squared + t_plus_one) * present_values) / price
    
    price_change_1bp = -modified_duration * price * 0.0001
    price_change_100bp = -modified_duration * price * 0.01
    convexity_adjustment_100bp = 0.5 * convexity * price * (0.01) ** 2
    
    result = {
        "macaulay_duration": macaulay_duration,
        "modified_duration": modified_duration,
        "convexity": convexity,
        "price": price,
        "face_value": face_value,
        "coupon_rate": coupon_rate,
        "years_to_maturity": years_to_maturity,
        "yield_to_maturity": yield_to_maturity,
        "frequency": frequency,
        "price_change_1bp": price_change_1bp,
        "price_change_100bp": price_change_100bp,
        "convexity_adjustment_100bp": convexity_adjustment_100bp,
        "price_change_100bp_with_convexity": price_change_100bp + convexity_adjustment_100bp
    }
    
    return result 