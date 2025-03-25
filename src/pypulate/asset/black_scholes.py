"""
Black-Scholes option pricing model implementation.
"""

from typing import Dict, Any, Literal, Union, Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes(
    option_type: Literal['call', 'put'],
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate option price using the Black-Scholes model.
    
    Parameters
    ----------
    option_type : str
        Type of option ('call' or 'put')
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
    dividend_yield : float, optional
        Continuous dividend yield, by default 0.0
        
    Returns
    -------
    dict
        Option price and Greeks
        
    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'
        If underlying_price or strike_price is not positive
        If time_to_expiry is not positive
        If volatility is not positive
    """
    # Validate inputs
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be either 'call' or 'put'")
    if underlying_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    
    # Calculate d1 and d2
    d1 = (np.log(underlying_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Calculate option price
    if option_type == 'call':
        price = (underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - 
                 strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
    else:  # put option
        price = (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                 underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1))
    
    # Calculate Greeks
    # Delta - first derivative of option price with respect to underlying price
    if option_type == 'call':
        delta = np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
    else:
        delta = np.exp(-dividend_yield * time_to_expiry) * (norm.cdf(d1) - 1)
    
    # Gamma - second derivative of option price with respect to underlying price
    gamma = (np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1)) / (underlying_price * volatility * np.sqrt(time_to_expiry))
    
    # Theta - derivative of option price with respect to time to expiry
    if option_type == 'call':
        theta = (-underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * volatility / 
                 (2 * np.sqrt(time_to_expiry)) - 
                 risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) +
                 dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1))
    else:
        theta = (-underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * volatility / 
                 (2 * np.sqrt(time_to_expiry)) + 
                 risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                 dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1))
    
    # Vega - derivative of option price with respect to volatility
    vega = underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * np.sqrt(time_to_expiry)
    
    # Rho - derivative of option price with respect to risk-free rate
    if option_type == 'call':
        rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    
    # Return results
    result = {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "d1": d1,
        "d2": d2,
        "underlying_price": underlying_price,
        "strike_price": strike_price,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield
    }
    
    return result


def implied_volatility(
    option_type: Literal['call', 'put'],
    market_price: float,
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    precision: float = 0.0001,
    max_iterations: int = 100,
    initial_vol: float = 0.2
) -> Dict[str, Any]:
    """
    Calculate implied volatility from option market price using the Black-Scholes model.
    
    Parameters
    ----------
    option_type : str
        Type of option ('call' or 'put')
    market_price : float
        Market price of the option
    underlying_price : float
        Current price of the underlying asset
    strike_price : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    dividend_yield : float, optional
        Continuous dividend yield, by default 0.0
    precision : float, optional
        Desired precision for implied volatility, by default 0.0001
    max_iterations : int, optional
        Maximum number of iterations, by default 100
    initial_vol : float, optional
        Initial volatility guess, by default 0.2
        
    Returns
    -------
    dict
        Implied volatility and option details
        
    Examples
    --------
    >>> from pypulate.asset import implied_volatility
    >>> result = implied_volatility(
    ...     option_type='call',
    ...     market_price=10.5,
    ...     underlying_price=100,
    ...     strike_price=100,
    ...     time_to_expiry=1.0,
    ...     risk_free_rate=0.05
    ... )
    >>> print(f"Implied Volatility: {result['implied_volatility']:.2%}")
    Implied Volatility: 20.12%
    """
    # Validate inputs
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be either 'call' or 'put'")
    if market_price <= 0 or underlying_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    
    # Define the objective function for root finding
    def objective(volatility):
        bs_price = black_scholes(
            option_type=option_type,
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield
        )['price']
        return bs_price - market_price
    
    try:
        # Use Brent's method to find the implied volatility
        implied_vol = brentq(
            objective,
            a=0.001,  # Lower bound
            b=5.0,    # Upper bound
            xtol=precision
        )
        
        # Calculate option price and Greeks with the implied volatility
        option_details = black_scholes(
            option_type=option_type,
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=implied_vol,
            dividend_yield=dividend_yield
        )
        
        # Return results
        result = {
            "implied_volatility": implied_vol,
            "market_price": market_price,
            "calculated_price": option_details['price'],
            "price_difference": abs(option_details['price'] - market_price),
            "delta": option_details['delta'],
            "gamma": option_details['gamma'],
            "theta": option_details['theta'],
            "vega": option_details['vega'],
            "rho": option_details['rho']
        }
        
        return result
    
    except ValueError:
        # If root finding fails, return an error message
        return {
            "error": "Could not find implied volatility within bounds",
            "market_price": market_price,
            "underlying_price": underlying_price,
            "strike_price": strike_price,
            "time_to_expiry": time_to_expiry,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield
        } 