"""
Binomial tree option pricing model implementation.
"""

from typing import Dict, Any, Literal, Union, Optional
import numpy as np
from numpy.ma import masked_array


def binomial_tree(
    option_type: Literal['european_call', 'european_put', 'american_call', 'american_put'],
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    steps: int = 100,
    dividend_yield: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate option price using the binomial tree model.
    
    Parameters
    ----------
    option_type : str
        Type of option ('european_call', 'european_put', 'american_call', 'american_put')
    underlying_price : float
        Current price of the underlying asset
    strike_price : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    volatility : float
        Volatility of the underlying asset (annualized)
    steps : int, optional
        Number of time steps in the binomial tree, by default 100
    dividend_yield : float, optional
        Continuous dividend yield, by default 0.0
        
    Returns
    -------
    dict
        Option price and details
        
    Raises
    ------
    ValueError
        If option_type is not one of 'european_call', 'european_put', 'american_call', 'american_put'
        If underlying_price or strike_price is not positive
        If time_to_expiry is not positive
        If volatility is not positive
        If steps is not a positive integer
    """
    valid_option_types = ['european_call', 'european_put', 'american_call', 'american_put']
    if option_type not in valid_option_types:
        raise ValueError(f"option_type must be one of {valid_option_types}")
    if underlying_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    if steps <= 0 or not isinstance(steps, int):
        raise ValueError("Steps must be a positive integer")
    
    dt: float = time_to_expiry / steps
    
    u: float = np.exp(volatility * np.sqrt(dt))
    d: float = 1.0 / u
    
    p: float = (np.exp((risk_free_rate - dividend_yield) * dt) - d) / (u - d)
    discount_factor: float = np.exp(-risk_free_rate * dt)
    
    mask = np.triu(np.ones((steps + 1, steps + 1), dtype=bool))
    price_tree = np.zeros((steps + 1, steps + 1), dtype=np.float64)
    price_tree = np.ma.array(price_tree, mask=~mask)
    
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = underlying_price * (u ** (i - j)) * (d ** j)
    
    option_tree = np.ma.array(np.zeros_like(price_tree), mask=~mask)
    
    if option_type in ['european_call', 'american_call']:
        option_tree[:, -1] = np.maximum(0, price_tree[:, -1] - strike_price)
    else:  
        option_tree[:, -1] = np.maximum(0, strike_price - price_tree[:, -1])
    
    for i in range(steps - 1, -1, -1):
        expected_values = (p * option_tree[:-1, i + 1] + 
                         (1 - p) * option_tree[1:, i + 1])
        option_values = discount_factor * expected_values
        
        if option_type in ['american_call', 'american_put']:
            if option_type == 'american_call':
                exercise_values = np.maximum(0, price_tree[:-1, i] - strike_price)
            else: 
                exercise_values = np.maximum(0, strike_price - price_tree[:-1, i])
            option_values = np.maximum(option_values, exercise_values)
        
        option_tree[:-1, i] = option_values
    
    price: float = float(option_tree[0, 0])
    
    if steps > 1:
        delta: Optional[float] = (option_tree[0, 1] - option_tree[1, 1]) / (price_tree[0, 1] - price_tree[1, 1])
        
        if steps > 2:
            price_diff_1 = abs(price_tree[0, 2] - price_tree[1, 2])
            price_diff_2 = abs(price_tree[1, 2] - price_tree[2, 2])
            avg_price_diff = abs(price_tree[0, 2] - price_tree[2, 2]) / 2
            
            if avg_price_diff > 1e-10:  
                gamma: Optional[float] = abs(
                    (option_tree[0, 2] - option_tree[1, 2]) / price_diff_1 -
                    (option_tree[1, 2] - option_tree[2, 2]) / price_diff_2
                ) / avg_price_diff
            else:
                gamma = None
        else:
            gamma = None
    else:
        delta = None
        gamma = None
    
    early_exercise_optimal: bool = False
    if option_type in ['american_call', 'american_put']:
        if option_type == 'american_call':
            intrinsic_value = max(0, underlying_price - strike_price)
        else: 
            intrinsic_value = max(0, strike_price - underlying_price)
        early_exercise_optimal = abs(price - intrinsic_value) < 1e-10
    
    result = {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "underlying_price": underlying_price,
        "strike_price": strike_price,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "steps": steps,
        "dividend_yield": dividend_yield,
        "risk_neutral_probability": p,
        "early_exercise_optimal": early_exercise_optimal if option_type in ['american_call', 'american_put'] else None
    }
    
    return result 