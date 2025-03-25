"""
Monte Carlo option pricing model implementation.
"""

from typing import Dict, Any, Literal, Union, Optional, Callable, List
import numpy as np
from scipy.stats import norm


def monte_carlo_option_pricing(
    option_type: Literal['european_call', 'european_put', 'asian_call', 'asian_put', 'lookback_call', 'lookback_put'],
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    simulations: int = 10000,
    time_steps: int = 252,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Price options using Monte Carlo simulation.
    
    This function implements Monte Carlo simulation for pricing various types of options,
    including European, Asian, and lookback options. It also supports jump diffusion for
    modeling assets with sudden price jumps like cryptocurrencies.
    
    Parameters
    ----------
    option_type : str
        Type of option ('european_call', 'european_put', 'asian_call', 'asian_put', 'lookback_call', 'lookback_put')
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
    simulations : int, optional
        Number of Monte Carlo simulations, by default 10000
    time_steps : int, optional
        Number of time steps in each simulation, by default 252
    dividend_yield : float, optional
        Continuous dividend yield, by default 0.0
    antithetic : bool, optional
        Whether to use antithetic variates for variance reduction, by default True
    jump_intensity : float, optional
        Expected number of jumps per year (lambda in Poisson process), by default 0.0
    jump_mean : float, optional
        Mean of the jump size distribution, by default 0.0
    jump_std : float, optional
        Standard deviation of the jump size distribution, by default 0.0
    seed : int, optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    dict
        Option price and details
        
    Raises
    ------
    ValueError
        If option_type is not valid, or prices are non-positive, or time to expiry is non-positive, or volatility is non-positive, or number of simulations or time steps are non-positive, or jump intensity or jump standard deviation are negative
    """
    valid_option_types = ['european_call', 'european_put', 'asian_call', 'asian_put', 'lookback_call', 'lookback_put']
    if option_type not in valid_option_types:
        raise ValueError(f"option_type must be one of {valid_option_types}")
    if underlying_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    if simulations <= 0 or not isinstance(simulations, int):
        raise ValueError("Number of simulations must be a positive integer")
    if time_steps <= 0 or not isinstance(time_steps, int):
        raise ValueError("Number of time steps must be a positive integer")
    if jump_intensity < 0:
        raise ValueError("Jump intensity must be non-negative")
    if jump_std < 0:
        raise ValueError("Jump standard deviation must be non-negative")
    
    if seed is not None:
        np.random.seed(seed)
    
    dt = time_to_expiry / time_steps
    
   
    drift = risk_free_rate * dt
    
    if jump_intensity > 0:
        expected_jump_effect = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        jump_compensation = jump_intensity * expected_jump_effect * dt
        drift -= jump_compensation
        
    vol_sqrt_dt = volatility * np.sqrt(dt)
    
    actual_simulations = simulations // 2 if antithetic else simulations
    
    if antithetic and simulations % 2 != 0:
        simulations = simulations + 1
        actual_simulations = simulations // 2
   
    current_prices = np.full(simulations, underlying_price, dtype=np.float64)
    
    if option_type in ['asian_call', 'asian_put']:
        price_sum = np.zeros(simulations, dtype=np.float64)
    elif option_type in ['lookback_call', 'lookback_put']:
        min_prices = np.full(simulations, underlying_price, dtype=np.float64)
        max_prices = np.full(simulations, underlying_price, dtype=np.float64)
    
    if antithetic:
        z = np.random.normal(0, 1, (actual_simulations, time_steps))

        samples = np.vstack([z/np.sqrt(2), -z/np.sqrt(2)])
    else:
        samples = np.random.normal(0, 1, (simulations, time_steps))
    
    if jump_intensity > 0:
        lambda_dt = jump_intensity * dt
        if antithetic:
            jump_counts = np.random.poisson(lambda_dt, (actual_simulations, time_steps))
        else:
            jump_counts = np.random.poisson(lambda_dt, (simulations, time_steps))
    
    for t in range(time_steps):
        price_movement = np.exp(drift + vol_sqrt_dt * samples[:, t])
        
        if jump_intensity > 0:
            if antithetic:
                jumps_array = np.zeros(simulations, dtype=np.float64)
                for i in range(actual_simulations):
                    if jump_counts[i, t] > 0:
                        jump_sizes = np.random.normal(jump_mean, jump_std, jump_counts[i, t])
                        total_jump: float = np.sum(jump_sizes)
                        jumps_array[i] = total_jump
                        jumps_array[i + actual_simulations] = -total_jump
                
                price_movement *= np.exp(jumps_array)
            else:
                for i in range(simulations):
                    if jump_counts[i, t] > 0:
                        jump_sizes = np.random.normal(jump_mean, jump_std, jump_counts[i, t])
                        jumps: float = np.sum(jump_sizes)
                        price_movement[i] *= np.exp(jumps)
        
        current_prices *= price_movement
        
        if option_type in ['asian_call', 'asian_put']:
            price_sum += current_prices
        elif option_type in ['lookback_call', 'lookback_put']:
            np.minimum(min_prices, current_prices, out=min_prices)
            np.maximum(max_prices, current_prices, out=max_prices)
    
    if option_type == 'european_call':
        payoffs = np.maximum(0, current_prices - strike_price)
    elif option_type == 'european_put':
        payoffs = np.maximum(0, strike_price - current_prices)
    elif option_type == 'asian_call':
        average_prices = price_sum / time_steps
        payoffs = np.maximum(0, average_prices - strike_price)
    elif option_type == 'asian_put':
        average_prices = price_sum / time_steps
        payoffs = np.maximum(0, strike_price - average_prices)
    elif option_type == 'lookback_call':
        payoffs = current_prices - min_prices
    elif option_type == 'lookback_put':
        payoffs = max_prices - current_prices
    else:
        raise ValueError(f"Unsupported option type: {option_type}")
    
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    option_price = discount_factor * np.mean(payoffs)
    
    std_error = np.std(payoffs) / np.sqrt(len(payoffs))
    
    confidence_interval = (
        discount_factor * (np.mean(payoffs) - 1.96 * std_error),
        discount_factor * (np.mean(payoffs) + 1.96 * std_error)
    )
    
    result = {
        "price": option_price,
        "standard_error": discount_factor * std_error,
        "confidence_interval": confidence_interval,
        "underlying_price": underlying_price,
        "strike_price": strike_price,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "simulations": simulations,
        "time_steps": time_steps,
        "dividend_yield": dividend_yield,
        "antithetic": antithetic,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std
    }
    
    return result 


def price_action_monte_carlo(
    option_type: Literal['european_call', 'european_put', 'asian_call', 'asian_put', 'lookback_call', 'lookback_put'],
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    support_levels: List[float],
    resistance_levels: List[float],
    respect_level_strength: float = 0.7, 
    volatility_near_levels: float = 1.3, 
    simulations: int = 10000,
    time_steps: int = 252,
    dividend_yield: float = 0.0,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
    antithetic: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Price options using Monte Carlo simulation with price action considerations.
    
    This function extends the standard Monte Carlo simulation by incorporating
    technical analysis elements such as support and resistance levels. The price
    paths are adjusted to respect these levels, with increased volatility near
    levels and potential bounces or breakouts.
    
    Parameters
    ----------
    option_type : str
        Type of option ('european_call', 'european_put', 'asian_call', 'asian_put', 
        'lookback_call', 'lookback_put')
    underlying_price : float
        Current price of the underlying asset
    strike_price : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    volatility : float
        Base volatility of the underlying asset (annualized)
    support_levels : List[float]
        List of price levels that act as support (in ascending order)
    resistance_levels : List[float]
        List of price levels that act as resistance (in ascending order)
    respect_level_strength : float
        How strongly the price respects support/resistance levels (0-1)
        0 = no respect (standard Monte Carlo), 1 = strong respect
    volatility_near_levels : float
        Volatility multiplier when price is near support/resistance levels
    simulations : int
        Number of Monte Carlo simulations
    time_steps : int
        Number of time steps in each simulation
    dividend_yield : float
        Continuous dividend yield
    jump_intensity : float
        Expected number of jumps per year (lambda in Poisson process)
    jump_mean : float
        Mean of the jump size distribution
    jump_std : float
        Standard deviation of the jump size distribution
    antithetic : bool
        Whether to use antithetic variates for variance reduction
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing the option price, standard error, and other information
    
    Raises
    ------
    ValueError
        If prices are non-positive, or time to expiry is non-positive, or volatility is non-positive, or number of simulations or time steps are non-positive, or jump intensity or jump standard deviation are negative, or respect level strength is not between 0 and 1, or volatility multiplier is non-positive
    """
    if underlying_price <= 0 or strike_price <= 0:
        raise ValueError("Prices must be positive")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    if simulations <= 0 or time_steps <= 0:
        raise ValueError("Simulation parameters must be positive")
    if jump_intensity < 0 or jump_std < 0:
        raise ValueError("Jump parameters cannot be negative")
    if respect_level_strength < 0 or respect_level_strength > 1:
        raise ValueError("Respect level strength must be between 0 and 1")
    if volatility_near_levels <= 0:
        raise ValueError("Volatility multiplier must be positive")
    
    support_levels = sorted([s for s in support_levels if s < underlying_price])
    resistance_levels = sorted([r for r in resistance_levels if r > underlying_price])
    
    nearest_support = support_levels[-1] if support_levels else underlying_price * 0.8
    nearest_resistance = resistance_levels[0] if resistance_levels else underlying_price * 1.2
    
    if seed is not None:
        np.random.seed(seed)
    
    dt = time_to_expiry / time_steps
    
    jump_compensation = 0.0
    if jump_intensity > 0:
        expected_jump_size = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        jump_compensation = jump_intensity * expected_jump_size
    
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2 - jump_compensation) * dt
    vol_sqrt_dt = volatility * np.sqrt(dt)
    
    actual_simulations = simulations
    if antithetic:
        actual_simulations = simulations // 2
    
    effective_simulations = 2 * actual_simulations if antithetic else simulations
    
    random_samples = np.random.normal(0, 1, (actual_simulations, time_steps))
    if antithetic:
        random_samples = np.vstack([random_samples, -random_samples])
    
    rng_state = np.random.get_state()
    bounce_random = np.random.random((effective_simulations, time_steps))
    bounce_direction = np.random.random((effective_simulations, time_steps))
    bounce_strength = np.random.uniform(0.001, 0.005, (effective_simulations, time_steps))
    np.random.set_state(rng_state)
    
    prices = np.full((effective_simulations, time_steps + 1), underlying_price, dtype=np.float64)
    
    if option_type in ['asian_call', 'asian_put']:
        price_sum = np.zeros_like(prices[:, 0])
    elif option_type in ['lookback_call', 'lookback_put']:
        min_prices = np.full_like(prices[:, 0], underlying_price)
        max_prices = np.full_like(prices[:, 0], underlying_price)
    
    if jump_intensity > 0:
        lambda_dt = jump_intensity * dt
        jump_counts = np.random.poisson(lambda_dt, (len(prices), time_steps))
    
    for t in range(1, time_steps + 1):
        prices[:, t] = prices[:, t-1] * np.exp(drift + vol_sqrt_dt * random_samples[:, t-1])
        
        if jump_intensity > 0:
            for i in range(len(prices)):
                if jump_counts[i, t-1] > 0:
                    jump_sizes = np.random.normal(jump_mean, jump_std, jump_counts[i, t-1])
                    jumps: float = np.sum(jump_sizes)
                    prices[i, t] *= np.exp(jumps)
        
        current_prices = prices[:, t]
        
        support_distance = (current_prices - nearest_support) / current_prices
        resistance_distance = (nearest_resistance - current_prices) / current_prices
        
        near_support_mask = support_distance < 0.02
        near_resistance_mask = resistance_distance < 0.02
        
        if np.any(near_support_mask):
            support_indices = np.where(near_support_mask)[0]
            
            bounce_mask = bounce_random[support_indices, t-1] < respect_level_strength
            
            bounce_indices = support_indices[bounce_mask]
            if len(bounce_indices) > 0:
                direction_up = bounce_direction[bounce_indices, t-1] < 0.5
                
                up_indices = bounce_indices[direction_up]
                if len(up_indices) > 0:
                    prices[up_indices, t] *= (1 + bounce_strength[up_indices, t-1])
                
                down_indices = bounce_indices[~direction_up]
                if len(down_indices) > 0:
                    prices[down_indices, t] *= (1 - bounce_strength[down_indices, t-1])
            
            break_indices = support_indices[~bounce_mask]
            if len(break_indices) > 0:
                local_volatility = volatility * (1 + (volatility_near_levels - 1) * 0.5)
                extra_moves = np.random.normal(0, local_volatility * np.sqrt(dt), len(break_indices))
                prices[break_indices, t] *= np.exp(extra_moves - 0.5 * local_volatility**2 * dt)
        
        if np.any(near_resistance_mask):
            resistance_indices = np.where(near_resistance_mask)[0]
            
            reject_mask = bounce_random[resistance_indices, t-1] < respect_level_strength
            
            reject_indices = resistance_indices[reject_mask]
            if len(reject_indices) > 0:
                direction_up = bounce_direction[reject_indices, t-1] < 0.5
                
                up_indices = reject_indices[direction_up]
                if len(up_indices) > 0:
                    prices[up_indices, t] *= (1 + bounce_strength[up_indices, t-1])
                
                down_indices = reject_indices[~direction_up]
                if len(down_indices) > 0:
                    prices[down_indices, t] *= (1 - bounce_strength[down_indices, t-1])
            
            break_indices = resistance_indices[~reject_mask]
            if len(break_indices) > 0:
                local_volatility = volatility * (1 + (volatility_near_levels - 1) * 0.5)
                extra_moves = np.random.normal(0, local_volatility * np.sqrt(dt), len(break_indices))
                prices[break_indices, t] *= np.exp(extra_moves - 0.5 * local_volatility**2 * dt)
        
        if option_type in ['asian_call', 'asian_put']:
            price_sum += prices[:, t]
        elif option_type in ['lookback_call', 'lookback_put']:
            np.minimum(min_prices, prices[:, t], out=min_prices)
            np.maximum(max_prices, prices[:, t], out=max_prices)
    
    if option_type == 'european_call':
        payoffs = np.maximum(0, prices[:, -1] - strike_price)
    elif option_type == 'european_put':
        payoffs = np.maximum(0, strike_price - prices[:, -1])
    elif option_type == 'asian_call':
        average_prices = price_sum / time_steps
        payoffs = np.maximum(0, average_prices - strike_price)
    elif option_type == 'asian_put':
        average_prices = price_sum / time_steps
        payoffs = np.maximum(0, strike_price - average_prices)
    elif option_type == 'lookback_call':
        payoffs = prices[:, -1] - min_prices
    elif option_type == 'lookback_put':
        payoffs = max_prices - prices[:, -1]
    else:
        raise ValueError(f"Unsupported option type: {option_type}")
    
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    option_price = discount_factor * np.mean(payoffs)
    
    std_error = np.std(payoffs) / np.sqrt(len(payoffs))
    confidence_interval = (
        option_price - 1.96 * discount_factor * std_error,
        option_price + 1.96 * discount_factor * std_error
    )
    
    result = {
        "price": option_price,
        "standard_error": discount_factor * std_error,
        "confidence_interval": confidence_interval,
        "underlying_price": underlying_price,
        "strike_price": strike_price,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "simulations": simulations,
        "time_steps": time_steps,
        "dividend_yield": dividend_yield,
        "antithetic": antithetic,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "respect_level_strength": respect_level_strength,
        "volatility_near_levels": volatility_near_levels
    }
    
    return result


def hybrid_price_action_monte_carlo(
    option_type: Literal['european_call', 'european_put', 'asian_call', 'asian_put', 'lookback_call', 'lookback_put'],
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    support_levels: List[float],
    resistance_levels: List[float],
    mean_reversion_params: Optional[Dict[str, float]] = None,
    jump_params: Optional[Dict[str, float]] = None,
    price_action_weight: float = 0.4,
    mean_reversion_weight: float = 0.3,
    jump_diffusion_weight: float = 0.3,
    respect_level_strength: float = 0.7,
    volatility_near_levels: float = 1.3,
    simulations: int = 10000,
    time_steps: int = 252,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    A hybrid option pricing model that combines price action Monte Carlo with
    mean reversion and jump diffusion models.
    
    This function creates a weighted average of three pricing models:
    1. Price action Monte Carlo (respects support/resistance)
    2. Mean reversion (if parameters provided)
    3. Jump diffusion (if parameters provided)
    
    Parameters
    ----------
    option_type : str
        Type of option ('european_call', 'european_put', 'asian_call', 'asian_put', 
        'lookback_call', 'lookback_put')
    underlying_price : float
        Current price of the underlying asset
    strike_price : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    volatility : float
        Base volatility of the underlying asset (annualized)
    support_levels : List[float]
        List of price levels that act as support
    resistance_levels : List[float]
        List of price levels that act as resistance
    mean_reversion_params : Dict, optional
        Parameters for mean reversion model: {'long_term_mean': float, 'mean_reversion_rate': float}
    jump_params : Dict, optional
        Parameters for jump diffusion: {'jump_intensity': float, 'jump_mean': float, 'jump_std': float}
    price_action_weight : float
        Weight for the price action model (0-1)
    mean_reversion_weight : float
        Weight for the mean reversion model (0-1)
    jump_diffusion_weight : float
        Weight for the jump diffusion model (0-1)
    respect_level_strength : float
        How strongly the price respects support/resistance levels (0-1)
    volatility_near_levels : float
        Volatility multiplier when price is near support/resistance levels
    simulations : int
        Number of Monte Carlo simulations
    time_steps : int
        Number of time steps in each simulation
    dividend_yield : float
        Continuous dividend yield
    antithetic : bool
        Whether to use antithetic variates for variance reduction
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing the option price, standard error, and other information
    
    Raises
    ------
    ValueError
        If weights do not sum to 1.0, or missing required parameters
    """
    if not np.isclose(price_action_weight + mean_reversion_weight + jump_diffusion_weight, 1.0, atol=1e-10):
        raise ValueError("Weights must sum to 1.0")
    
    if mean_reversion_params is None:
        mean_reversion_params = {
            'long_term_mean': underlying_price * 1.05, 
            'mean_reversion_rate': 1.0
        }
    
    if jump_params is None:
        jump_params = {
            'jump_intensity': 0.0,
            'jump_mean': 0.0,
            'jump_std': 0.0
        }
    
    required_mean_reversion = ['long_term_mean', 'mean_reversion_rate']
    required_jump = ['jump_intensity', 'jump_mean', 'jump_std']
    
    for param in required_mean_reversion:
        if param not in mean_reversion_params:
            raise ValueError(f"Missing required mean reversion parameter: {param}")
    
    for param in required_jump:
        if param not in jump_params:
            raise ValueError(f"Missing required jump parameter: {param}")
    
    if seed is not None:
        np.random.seed(seed)
        random_seed = seed
    else:
        random_seed = None
    
    model_simulations = max(1000, simulations // 3)
    
    long_term_mean = mean_reversion_params['long_term_mean']
    mean_reversion_rate = mean_reversion_params['mean_reversion_rate']
    mean_reversion_volatility = volatility * np.exp(-mean_reversion_rate * time_to_expiry / 2)
    
    results = {}
    models_to_run = []
    
    if price_action_weight > 0:
        models_to_run.append('price_action')
    if mean_reversion_weight > 0:
        models_to_run.append('mean_reversion')
    if jump_diffusion_weight > 0:
        models_to_run.append('jump_diffusion')
    
    if 'price_action' in models_to_run:
        price_action_result = price_action_monte_carlo(
            option_type=option_type,
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            respect_level_strength=respect_level_strength,
            volatility_near_levels=volatility_near_levels,
            simulations=model_simulations,
            time_steps=time_steps,
            dividend_yield=dividend_yield,
            jump_intensity=0.0, 
            jump_mean=0.0,
            jump_std=0.0,
            antithetic=antithetic,
            seed=random_seed
        )
        results['price_action'] = price_action_result
    
    if 'jump_diffusion' in models_to_run:
        jump_result = monte_carlo_option_pricing(
            option_type=option_type,
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            simulations=model_simulations,
            time_steps=time_steps,
            dividend_yield=dividend_yield,
            jump_intensity=jump_params['jump_intensity'],
            jump_mean=jump_params['jump_mean'],
            jump_std=jump_params['jump_std'],
            antithetic=antithetic,
            seed=random_seed
        )
        results['jump_diffusion'] = jump_result
    
    if 'mean_reversion' in models_to_run:
        mean_reversion_result = price_action_monte_carlo(
            option_type=option_type,
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=mean_reversion_volatility,
            support_levels=[long_term_mean * 0.95],  
            resistance_levels=[long_term_mean * 1.05], 
            respect_level_strength=mean_reversion_rate / 10, 
            volatility_near_levels=1.0, 
            simulations=model_simulations,
            time_steps=time_steps,
            dividend_yield=dividend_yield,
            jump_intensity=0.0, 
            jump_mean=0.0,
            jump_std=0.0,
            antithetic=antithetic,
            seed=random_seed
        )
        results['mean_reversion'] = mean_reversion_result
    
    hybrid_price = 0.0
    hybrid_std_error_squared = 0.0
    
    if 'price_action' in results:
        hybrid_price += price_action_weight * results['price_action']['price']
        hybrid_std_error_squared += (price_action_weight * results['price_action']['standard_error'])**2
    
    if 'mean_reversion' in results:
        hybrid_price += mean_reversion_weight * results['mean_reversion']['price']
        hybrid_std_error_squared += (mean_reversion_weight * results['mean_reversion']['standard_error'])**2
    
    if 'jump_diffusion' in results:
        hybrid_price += jump_diffusion_weight * results['jump_diffusion']['price']
        hybrid_std_error_squared += (jump_diffusion_weight * results['jump_diffusion']['standard_error'])**2
    
    hybrid_std_error = np.sqrt(hybrid_std_error_squared)
    
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    confidence_interval = (
        hybrid_price - 1.96 * hybrid_std_error,
        hybrid_price + 1.96 * hybrid_std_error
    )
    
    result = {
        "price": hybrid_price,
        "standard_error": hybrid_std_error,
        "confidence_interval": confidence_interval,
        "underlying_price": underlying_price,
        "strike_price": strike_price,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "simulations": model_simulations * len(models_to_run), 
        "time_steps": time_steps,
        "dividend_yield": dividend_yield,
        "antithetic": antithetic,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "price_action_weight": price_action_weight,
        "mean_reversion_weight": mean_reversion_weight,
        "jump_diffusion_weight": jump_diffusion_weight,
        "mean_reversion_params": mean_reversion_params,
        "jump_params": jump_params
    }
    
    if 'price_action' in results:
        result["price_action_price"] = results['price_action']['price']
    
    if 'mean_reversion' in results:
        result["mean_reversion_price"] = results['mean_reversion']['price']
    
    if 'jump_diffusion' in results:
        result["jump_diffusion_price"] = results['jump_diffusion']['price']
    
    return result 