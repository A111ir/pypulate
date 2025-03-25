"""
Term structure models for yield curve fitting.
"""

from typing import Dict, Any, List, Union, Optional, Tuple, Callable
import numpy as np
from scipy.optimize import minimize


def nelson_siegel(
    maturities: List[float],
    rates: List[float],
    initial_params: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Fit the Nelson-Siegel model to yield curve data.
    
    The Nelson-Siegel model is defined as:
    r(t) = β₀ + β₁ * (1 - exp(-t/τ))/(t/τ) + β₂ * ((1 - exp(-t/τ))/(t/τ) - exp(-t/τ))
    
    Parameters
    ----------
    maturities : list of float
        Maturities in years for the observed rates
    rates : list of float
        Observed interest rates (as decimals)
    initial_params : list of float, optional
        Initial parameters [β₀, β₁, β₂, τ], by default [0.03, -0.02, -0.01, 1.5]
        
    Returns
    -------
    dict
        Fitted parameters and model details
        
    Examples
    --------
    >>> from pypulate.asset import nelson_siegel
    >>> result = nelson_siegel(
    ...     maturities=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    ...     rates=[0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    ... )
    >>> # Get the fitted parameters
    >>> beta0, beta1, beta2, tau = result['parameters']
    >>> print(f"Long-term rate (β₀): {beta0:.2%}")
    Long-term rate (β₀): 3.40%
    >>> # Predict rate at a specific maturity
    >>> rate_4y = result['predict_func'](4)
    >>> print(f"4-year rate: {rate_4y:.2%}")
    4-year rate: 2.93%
    """
    # Validate inputs
    if len(maturities) != len(rates):
        raise ValueError("Length of maturities and rates must be the same")
    if len(maturities) < 4:
        raise ValueError("At least four points are required to fit the Nelson-Siegel model")
    if not all(m > 0 for m in maturities):
        raise ValueError("All maturities must be positive")
    if not all(r >= 0 for r in rates):
        raise ValueError("All rates must be non-negative")
    
    # Default initial parameters if not provided
    if initial_params is None:
        initial_params = [0.03, -0.02, -0.01, 1.5]  # [β₀, β₁, β₂, τ]
    
    if len(initial_params) != 4:
        raise ValueError("initial_params must have exactly 4 elements [β₀, β₁, β₂, τ]")
    
    # Convert inputs to numpy arrays
    maturities_array = np.array(maturities)
    rates_array = np.array(rates)
    
    # Define the Nelson-Siegel function
    def ns_function(t, params):
        beta0, beta1, beta2, tau = params
        
        # Handle very small maturities to avoid division by zero
        if isinstance(t, (int, float)) and t < 1e-10:
            t = 1e-10
        elif isinstance(t, np.ndarray):
            t = np.maximum(t, 1e-10)
        
        # Calculate the components
        exp_term = np.exp(-t / tau)
        term1 = (1 - exp_term) / (t / tau)
        term2 = term1 - exp_term
        
        # Calculate the rate
        rate = beta0 + beta1 * term1 + beta2 * term2
        
        return rate
    
    # Define the objective function to minimize (sum of squared errors)
    def objective(params):
        predicted = ns_function(maturities_array, params)
        return np.sum((predicted - rates_array) ** 2)
    
    # Add constraints to ensure tau is positive
    constraints = [{'type': 'ineq', 'fun': lambda params: params[3]}]
    
    # Fit the model using optimization
    result = minimize(objective, initial_params, method='SLSQP', constraints=constraints)
    
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    # Extract fitted parameters
    fitted_params = result.x
    beta0, beta1, beta2, tau = fitted_params
    
    # Create prediction function
    def predict_func(t):
        return ns_function(t, fitted_params)
    
    # Calculate fitted rates
    fitted_rates = predict_func(maturities_array)
    
    # Calculate goodness of fit
    residuals = rates_array - fitted_rates
    sse: float = np.sum(residuals ** 2)
    sst: float = np.sum((rates_array - np.mean(rates_array)) ** 2)
    r_squared = 1 - (sse / sst) if sst > 0 else 0
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Calculate asymptotic values
    short_rate = predict_func(1e-10)  # Very short-term rate
    long_rate = beta0  # Long-term rate is β₀
    
    # Return results
    return {
        "parameters": fitted_params.tolist(),
        "parameter_names": ["beta0", "beta1", "beta2", "tau"],
        "predict_func": predict_func,
        "fitted_rates": fitted_rates.tolist(),
        "residuals": residuals.tolist(),
        "r_squared": r_squared,
        "rmse": rmse,
        "short_rate": short_rate,
        "long_rate": long_rate,
        "maturities": maturities,
        "rates": rates
    }


def svensson(
    maturities: List[float],
    rates: List[float],
    initial_params: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Fit the Svensson model to yield curve data.
    
    The Svensson model is defined as:
    r(t) = β₀ + β₁ * (1 - exp(-t/τ₁))/(t/τ₁) + 
           β₂ * ((1 - exp(-t/τ₁))/(t/τ₁) - exp(-t/τ₁)) +
           β₃ * ((1 - exp(-t/τ₂))/(t/τ₂) - exp(-t/τ₂))
    
    Parameters
    ----------
    maturities : list of float
        Maturities in years for the observed rates
    rates : list of float
        Observed interest rates (as decimals)
    initial_params : list of float, optional
        Initial parameters [β₀, β₁, β₂, β₃, τ₁, τ₂], by default [0.03, -0.02, -0.01, 0.01, 1.5, 10]
        
    Returns
    -------
    dict
        Fitted parameters and model details
        
    Examples
    --------
    >>> from pypulate.asset import svensson
    >>> result = svensson(
    ...     maturities=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    ...     rates=[0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    ... )
    >>> # Get the fitted parameters
    >>> beta0, beta1, beta2, beta3, tau1, tau2 = result['parameters']
    >>> print(f"Long-term rate (β₀): {beta0:.2%}")
    Long-term rate (β₀): 3.40%
    >>> # Predict rate at a specific maturity
    >>> rate_4y = result['predict_func'](4)
    >>> print(f"4-year rate: {rate_4y:.2%}")
    4-year rate: 2.93%
    """
    # Validate inputs
    if len(maturities) != len(rates):
        raise ValueError("Length of maturities and rates must be the same")
    if len(maturities) < 6:
        raise ValueError("At least six points are required to fit the Svensson model")
    if not all(m > 0 for m in maturities):
        raise ValueError("All maturities must be positive")
    if not all(r >= 0 for r in rates):
        raise ValueError("All rates must be non-negative")
    
    # Default initial parameters if not provided
    if initial_params is None:
        initial_params = [0.03, -0.02, -0.01, 0.01, 1.5, 10]  # [β₀, β₁, β₂, β₃, τ₁, τ₂]
    
    if len(initial_params) != 6:
        raise ValueError("initial_params must have exactly 6 elements [β₀, β₁, β₂, β₃, τ₁, τ₂]")
    
    # Convert inputs to numpy arrays
    maturities_array = np.array(maturities)
    rates_array = np.array(rates)
    
    # Define the Svensson function
    def svensson_function(t, params):
        beta0, beta1, beta2, beta3, tau1, tau2 = params
        
        # Handle very small maturities to avoid division by zero
        if isinstance(t, (int, float)) and t < 1e-10:
            t = 1e-10
        elif isinstance(t, np.ndarray):
            t = np.maximum(t, 1e-10)
        
        # Calculate the components for τ₁
        exp_term1 = np.exp(-t / tau1)
        term1 = (1 - exp_term1) / (t / tau1)
        term2 = term1 - exp_term1
        
        # Calculate the components for τ₂
        exp_term2 = np.exp(-t / tau2)
        term3 = (1 - exp_term2) / (t / tau2)
        term4 = term3 - exp_term2
        
        # Calculate the rate
        rate = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term4
        
        return rate
    
    # Define the objective function to minimize (sum of squared errors)
    def objective(params):
        predicted = svensson_function(maturities_array, params)
        return np.sum((predicted - rates_array) ** 2)
    
    # Add constraints to ensure tau values are positive
    constraints = [
        {'type': 'ineq', 'fun': lambda params: params[4]},  # τ₁ > 0
        {'type': 'ineq', 'fun': lambda params: params[5]}   # τ₂ > 0
    ]
    
    # Fit the model using optimization
    result = minimize(objective, initial_params, method='SLSQP', constraints=constraints)
    
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    # Extract fitted parameters
    fitted_params = result.x
    beta0, beta1, beta2, beta3, tau1, tau2 = fitted_params
    
    # Create prediction function
    def predict_func(t):
        return svensson_function(t, fitted_params)
    
    # Calculate fitted rates
    fitted_rates = predict_func(maturities_array)
    
    # Calculate goodness of fit
    residuals = rates_array - fitted_rates
    sse: float = np.sum(residuals ** 2)
    sst: float = np.sum((rates_array - np.mean(rates_array)) ** 2)
    r_squared = 1 - (sse / sst) if sst > 0 else 0
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Calculate asymptotic values
    short_rate = predict_func(1e-10)  # Very short-term rate
    long_rate = beta0  # Long-term rate is β₀
    
    # Return results
    return {
        "parameters": fitted_params.tolist(),
        "parameter_names": ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"],
        "predict_func": predict_func,
        "fitted_rates": fitted_rates.tolist(),
        "residuals": residuals.tolist(),
        "r_squared": r_squared,
        "rmse": rmse,
        "short_rate": short_rate,
        "long_rate": long_rate,
        "maturities": maturities,
        "rates": rates
    } 