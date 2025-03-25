"""
Mean inversion pricing model for asset pricing.

This module implements the Ornstein-Uhlenbeck process for mean-reverting assets
such as commodities, interest rates, and volatility.
"""

from typing import Dict, Any, List, Union, Optional, Tuple
import numpy as np
from scipy.stats import norm


def mean_inversion_pricing(
    current_price: float,
    long_term_mean: float,
    mean_reversion_rate: float,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    strike_price: float,
    option_type: str = 'call',
    simulations: int = 10000,
    time_steps: int = 252,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Price options on mean-reverting assets using Monte Carlo simulation.
    
    This function implements the Ornstein-Uhlenbeck process to model mean-reverting
    assets and prices options using Monte Carlo simulation.
    
    Parameters
    ----------
    current_price : float
        Current price of the underlying asset
    long_term_mean : float
        Long-term mean level that the asset price reverts to
    mean_reversion_rate : float
        Speed at which the asset price reverts to the long-term mean (annualized)
    volatility : float
        Volatility of the asset price (annualized)
    time_to_expiry : float
        Time to option expiration in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    strike_price : float
        Strike price of the option
    option_type : str, optional
        Type of option ('call' or 'put'), by default 'call'
    simulations : int, optional
        Number of Monte Carlo simulations, by default 10000
    time_steps : int, optional
        Number of time steps in each simulation, by default 252
    seed : int, optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    dict
        Option price and details
        
    Raises
    ------
    ValueError
        If Current price, volatility, time to expiry, mean reversion rate, or simulations are not positive.
        If option type is not 'call' or 'put'.
        If time steps is not a positive integer.
    """
    if current_price <= 0:
        raise ValueError("Current price must be positive")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if mean_reversion_rate < 0:
        raise ValueError("Mean reversion rate must be non-negative")
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")
    if simulations <= 0 or not isinstance(simulations, int):
        raise ValueError("Simulations must be a positive integer")
    if time_steps <= 0 or not isinstance(time_steps, int):
        raise ValueError("Time steps must be a positive integer")
    
    if seed is not None:
        np.random.seed(seed)
    
    dt = time_to_expiry / time_steps
    theta = mean_reversion_rate
    mu = long_term_mean
    sigma = volatility
    
    vis_paths = np.zeros((5, time_steps + 1))
    vis_paths[:, 0] = current_price
    
    batch_size = min(10000, simulations)  
    n_batches = (simulations + batch_size - 1) // batch_size
    
    payoff_sum = 0.0
    payoff_squared_sum = 0.0
    
    for batch in range(n_batches):
        current_batch_size = min(batch_size, simulations - batch * batch_size)
        
        prices = np.full(current_batch_size, current_price)
        
        if theta > 0:
            for t in range(time_steps):
                mean = mu + (prices - mu) * np.exp(-theta * dt)
                var = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
                prices = np.random.normal(mean, np.sqrt(var))
                prices = np.maximum(prices, 0) 
                
                if batch == 0 and t < time_steps:
                    vis_paths[:min(5, current_batch_size), t+1] = prices[:min(5, current_batch_size)]
        else:
            for t in range(time_steps):
                prices = np.random.normal(prices, sigma * np.sqrt(dt))
                prices = np.maximum(prices, 0) 
                
                if batch == 0 and t < time_steps:
                    vis_paths[:min(5, current_batch_size), t+1] = prices[:min(5, current_batch_size)]
        
        payoffs = np.maximum(prices - strike_price, 0) if option_type == 'call' else np.maximum(strike_price - prices, 0)
        
        payoff_sum += np.sum(payoffs)
        payoff_squared_sum += np.sum(payoffs**2)
    
    mean_payoff = payoff_sum / simulations
    mean_squared_payoff = payoff_squared_sum / simulations
    variance_payoff = mean_squared_payoff - mean_payoff**2
    
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    option_price = discount_factor * mean_payoff
    std_error = np.sqrt(variance_payoff / simulations)
    
    confidence_interval = (
        discount_factor * (mean_payoff - 1.96 * std_error),
        discount_factor * (mean_payoff + 1.96 * std_error)
    )
    
    if theta > 0:
        expected_price = mu + (current_price - mu) * np.exp(-theta * time_to_expiry)
    else:
        expected_price = current_price
    
    half_life = np.log(2) / theta if theta > 0 else float('inf')
    
    result = {
        "price": option_price,
        "standard_error": discount_factor * std_error,
        "confidence_interval": confidence_interval,
        "current_price": current_price,
        "long_term_mean": long_term_mean,
        "mean_reversion_rate": mean_reversion_rate,
        "volatility": volatility,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "strike_price": strike_price,
        "option_type": option_type,
        "simulations": simulations,
        "time_steps": time_steps,
        "expected_price_at_expiry": expected_price,
        "half_life": half_life,
        "price_statistics": {
            "mean": expected_price,  # Using theoretical mean
            "std": np.sqrt((sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * time_to_expiry))) if theta > 0 else sigma * np.sqrt(time_to_expiry),
            "min": np.min(prices) if current_batch_size > 0 else current_price,
            "max": np.max(prices) if current_batch_size > 0 else current_price,
            "median": np.median(prices) if current_batch_size > 0 else current_price
        },
        "sample_paths": vis_paths.tolist() 
    }
    
    return result


def analytical_mean_inversion_option(
    current_price: float,
    long_term_mean: float,
    mean_reversion_rate: float,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    strike_price: float,
    option_type: str = 'call'
) -> Dict[str, Any]:
    """
    Price European options on mean-reverting assets using an analytical approximation.
    
    This function implements an analytical approximation for pricing European options
    on mean-reverting assets based on the Ornstein-Uhlenbeck process.
    
    Parameters
    ----------
    current_price : float
        Current price of the underlying asset
    long_term_mean : float
        Long-term mean level that the asset price reverts to
    mean_reversion_rate : float
        Speed at which the asset price reverts to the long-term mean (annualized)
    volatility : float
        Volatility of the asset price (annualized)
    time_to_expiry : float
        Time to option expiration in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    strike_price : float
        Strike price of the option
    option_type : str, optional
        Type of option ('call' or 'put'), by default 'call'
        
    Returns
    -------
    dict
        Option price and details
        
    Raises
    ------
    ValueError
        If Current price, volatility, time to expiry, mean reversion rate, or strike price are not positive.
        If option type is not 'call' or 'put'.
    """
    if current_price <= 0:
        raise ValueError("Current price must be positive")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if mean_reversion_rate < 0:
        raise ValueError("Mean reversion rate must be non-negative")
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")
    
    theta = mean_reversion_rate
    mu = long_term_mean
    sigma = volatility
    t = time_to_expiry
    r = risk_free_rate
    
    if theta > 0:
        expected_price = mu + (current_price - mu) * np.exp(-theta * t)
        variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))
    else:
        expected_price = current_price
        variance = sigma**2 * t
    
    std_dev = np.sqrt(variance)
    
    risk_adjusted_mean = current_price * np.exp(r * t)
    
    d1 = (np.log(risk_adjusted_mean / strike_price) + 0.5 * variance) / std_dev
    d2 = d1 - std_dev
    
    discount_factor = np.exp(-r * t)
    
    if option_type == 'call':
        option_price = discount_factor * (risk_adjusted_mean * norm.cdf(d1) - strike_price * norm.cdf(d2))
    else: 
        option_price = discount_factor * (strike_price * norm.cdf(-d2) - risk_adjusted_mean * norm.cdf(-d1))
    
    half_life = np.log(2) / theta if theta > 0 else float('inf')
    
    result = {
        "price": option_price,
        "current_price": current_price,
        "long_term_mean": long_term_mean,
        "mean_reversion_rate": mean_reversion_rate,
        "volatility": volatility,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "strike_price": strike_price,
        "option_type": option_type,
        "expected_price_at_expiry": expected_price,
        "variance_at_expiry": variance,
        "std_dev_at_expiry": std_dev,
        "half_life": half_life,
        "d1": d1,
        "d2": d2
    }
    
    return result 