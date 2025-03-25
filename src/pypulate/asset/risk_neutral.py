"""
Risk-neutral valuation for derivatives pricing.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np


def risk_neutral_valuation(
    payoff_function: Callable[[float], float],
    underlying_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    steps: int = 100,
    simulations: int = 1000,
    dividend_yield: float = 0.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Price a derivative using risk-neutral valuation with optimized NumPy vectorization.
    
    Parameters
    ----------
    payoff_function : callable
        Function that takes the final underlying price and returns the payoff
    underlying_price : float
        Current price of the underlying asset
    risk_free_rate : float
        Risk-free interest rate (annualized)
    volatility : float
        Volatility of the underlying asset (annualized)
    time_to_expiry : float
        Time to expiration in years
    steps : int, optional
        Number of time steps in the simulation, by default 100
    simulations : int, optional
        Number of Monte Carlo simulations, by default 1000
    dividend_yield : float, optional
        Continuous dividend yield, by default 0.0
    seed : int, optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    dict
        Derivative price and details
        
    Raises
    ------
    ValueError
        If the payoff function is not callable, underlying price is not positive, time to expiry is not positive, volatility is not positive, steps is not a positive integer, or simulations is not a positive integer.
    """
    # Input validation
    if not callable(payoff_function):
        raise ValueError("payoff_function must be callable")
    if underlying_price <= 0:
        raise ValueError("Underlying price must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    if steps <= 0 or not isinstance(steps, int):
        raise ValueError("Steps must be a positive integer")
    if simulations <= 0 or not isinstance(simulations, int):
        raise ValueError("Simulations must be a positive integer")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Precompute constants
    dt = time_to_expiry / steps
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * dt
    vol_sqrt_dt = volatility * np.sqrt(dt)
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    
    # Generate all random samples at once (more efficient than generating in the loop)
    random_samples = np.random.normal(0, 1, (simulations, steps))
    
    # Initialize price paths array with explicit dtype for better performance
    # We only need to track the current prices, not the full path history
    current_prices = np.full(simulations, underlying_price, dtype=np.float64)
    
    # Simulate price paths using vectorized operations
    # This is more memory efficient than storing the full price history
    for t in range(steps):
        # Update all prices in one vectorized operation
        current_prices *= np.exp(drift + vol_sqrt_dt * random_samples[:, t])
    
    # Apply payoff function to final prices
    # Use np.vectorize for clean syntax, but note that it's not always faster than a loop
    # For simple payoffs, this is efficient enough
    vectorized_payoff = np.vectorize(payoff_function)
    payoffs = vectorized_payoff(current_prices)
    
    # Calculate option price and statistics in a vectorized way
    payoff_mean = np.mean(payoffs)
    payoff_std = np.std(payoffs)
    std_error = payoff_std / np.sqrt(simulations)
    
    # Calculate derivative price
    derivative_price = discount_factor * payoff_mean
    
    # Calculate confidence interval (99% confidence)
    confidence_interval = (
        discount_factor * (payoff_mean - 2.58 * std_error),
        discount_factor * (payoff_mean + 2.58 * std_error)
    )
    
    # Calculate price statistics
    price_stats = {
        "mean": np.mean(current_prices),
        "std": np.std(current_prices),
        "min": np.min(current_prices),
        "max": np.max(current_prices),
        "median": np.median(current_prices)
    }
    
    # Calculate payoff statistics
    payoff_stats = {
        "mean": payoff_mean,
        "std": payoff_std,
        "min": np.min(payoffs),
        "max": np.max(payoffs),
        "median": np.median(payoffs),
        "zero_proportion": np.sum(payoffs == 0) / simulations
    }
    
    # Return comprehensive results
    result = {
        "price": derivative_price,
        "standard_error": discount_factor * std_error,
        "confidence_interval": confidence_interval,
        "underlying_price": underlying_price,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "time_to_expiry": time_to_expiry,
        "steps": steps,
        "simulations": simulations,
        "dividend_yield": dividend_yield,
        "price_statistics": price_stats,
        "payoff_statistics": payoff_stats
    }
    
    return result 