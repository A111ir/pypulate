"""
Risk-adjusted performance measurement functions for portfolio analysis.

This module provides functions for measuring risk-adjusted performance metrics
including Sharpe ratio, Information ratio, CAPM alpha, and multifactor models.

All functions support both Python lists and NumPy arrays as inputs.
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from scipy import stats


def sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: Union[float, List[float], np.ndarray] = 0.0,
    annualization_factor: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Calculate the Sharpe ratio, which measures excess return per unit of risk.
    
    Parameters
    ----------
    returns : array-like
        Array of periodic returns
    risk_free_rate : float or array-like, default 0.0
        Risk-free rate for the same period as returns
    annualization_factor : float, default 1.0
        Factor to annualize the Sharpe ratio (e.g., 252 for daily returns to annual)
        
    Returns
    -------
    float or ndarray
        The Sharpe ratio
        If array input is provided for risk_free_rate, returns an array of Sharpe ratios
        
    Examples
    --------
    >>> sharpe_ratio([0.01, 0.02, -0.01, 0.03, 0.01], 0.001, 252)
    2.5298221281347035
    >>> sharpe_ratio([0.01, 0.02, -0.01, 0.03, 0.01], [0.001, 0.002], 252)
    array([2.52982213, 2.26684001])
    """
    returns = np.asarray(returns)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate = np.asarray(risk_free_rate)
        
    excess_returns = returns.mean() - risk_free_rate
    
    volatility = returns.std()
    
    sharpe = excess_returns / volatility
    
    if annualization_factor != 1.0:
        sharpe = sharpe * np.sqrt(annualization_factor)
        
    return sharpe


def information_ratio(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray],
    annualization_factor: float = 1.0
) -> float:
    """
    Calculate the Information ratio, which measures active return per unit of active risk.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    benchmark_returns : array-like
        Array of benchmark returns for the same periods
    annualization_factor : float, default 1.0
        Factor to annualize the Information ratio (e.g., 252 for daily returns to annual)
        
    Returns
    -------
    float
        The Information ratio
        
    Examples
    --------
    >>> information_ratio([0.01, 0.02, -0.01, 0.03, 0.01], [0.005, 0.01, -0.005, 0.02, 0.005], 252)
    2.8284271247461903
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    
    active_returns = returns - benchmark_returns
    
    active_return = active_returns.mean()
    
    tracking_error = active_returns.std()
    
    ir = active_return / tracking_error
    
    if annualization_factor != 1.0:
        ir = ir * np.sqrt(annualization_factor)
        
    return ir


def capm_alpha(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray],
    risk_free_rate: Union[float, List[float], np.ndarray] = 0.0
) -> Tuple[float, float, float, float, float]:
    """
    Calculate the CAPM alpha (Jensen's alpha) and related statistics.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    benchmark_returns : array-like
        Array of benchmark returns for the same periods
    risk_free_rate : float or array-like, default 0.0
        Risk-free rate for the same period as returns
        
    Returns
    -------
    tuple
        (alpha, beta, r_squared, p_value, std_err)
        - alpha: The CAPM alpha (intercept)
        - beta: The CAPM beta (slope)
        - r_squared: The R-squared of the regression
        - p_value: The p-value for alpha
        - std_err: The standard error of alpha
        
    Examples
    --------
    >>> capm_alpha([0.01, 0.02, -0.01, 0.03, 0.01], [0.005, 0.01, -0.005, 0.02, 0.005], 0.001)
    (0.0046, 1.2, 0.9, 0.0023, 0.0012) 
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate = np.asarray(risk_free_rate)
        risk_free_rate = risk_free_rate.mean()
    
    excess_returns = returns - risk_free_rate
    excess_benchmark_returns = benchmark_returns - risk_free_rate
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        excess_benchmark_returns, excess_returns
    )
    
    alpha = intercept
    beta = slope
    r_squared = r_value ** 2
    
    return alpha, beta, r_squared, p_value, std_err


def benchmark_alpha(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate the benchmark alpha, which is the difference between portfolio return
    and benchmark return.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    benchmark_returns : array-like
        Array of benchmark returns for the same periods
        
    Returns
    -------
    float
        The benchmark alpha (difference in mean returns)
        
    Examples
    --------
    >>> benchmark_alpha([0.01, 0.02, -0.01, 0.03, 0.01], [0.005, 0.01, -0.005, 0.02, 0.005])
    0.005
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    
    portfolio_mean_return = returns.mean()
    benchmark_mean_return = benchmark_returns.mean()
    
    alpha = portfolio_mean_return - benchmark_mean_return
    
    return alpha


def multifactor_alpha(
    returns: Union[List[float], np.ndarray],
    factor_returns: Union[List[List[float]], np.ndarray],
    risk_free_rate: Union[float, List[float], np.ndarray] = 0.0
) -> Tuple[float, np.ndarray, float, float, float]:
    """
    Calculate the alpha from a multifactor model (e.g., Fama-French).
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    factor_returns : array-like
        2D array where each column represents returns for a factor
    risk_free_rate : float or array-like, default 0.0
        Risk-free rate for the same period as returns
        
    Returns
    -------
    tuple
        (alpha, betas, r_squared, p_value, std_err)
        - alpha: The multifactor alpha (intercept)
        - betas: Array of factor betas (coefficients)
        - r_squared: The R-squared of the regression
        - p_value: The p-value for alpha
        - std_err: The standard error of alpha
        
    Examples
    --------
    >>> # Example with market, size, and value factors
    >>> portfolio_returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    >>> factor_returns = [
    ...     [0.005, 0.01, -0.005, 0.02, 0.005],  # Market
    ...     [0.002, 0.003, -0.001, 0.004, 0.001],  # Size
    ...     [0.001, 0.002, -0.002, 0.003, 0.002]   # Value
    ... ]
    >>> multifactor_alpha(portfolio_returns, factor_returns, 0.001)
    (0.0032, array([0.9, 0.5, 0.3]), 0.92, 0.04, 0.0015)  # Example values
    """
    returns = np.asarray(returns)
    factor_returns = np.asarray(factor_returns)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate = np.asarray(risk_free_rate)
        risk_free_rate = risk_free_rate.mean()
    
    excess_returns = returns - risk_free_rate
    
    if factor_returns.shape[0] == len(returns) and factor_returns.ndim > 1:
        X = factor_returns 
    else:
        X = factor_returns.T 
    
    X_with_const = np.column_stack([np.ones(len(excess_returns)), X])
    
    result = np.linalg.lstsq(X_with_const, excess_returns, rcond=None)
    coefficients = result[0]
    
    alpha = coefficients[0]
    betas = coefficients[1:]
    
    y_pred = X_with_const @ coefficients
    ss_total = np.sum((excess_returns - np.mean(excess_returns))**2)
    ss_residual = np.sum((excess_returns - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    
    n = len(excess_returns)
    k = len(betas)
    degrees_of_freedom = n - k - 1
    
    if degrees_of_freedom > 0:
        mse = ss_residual / degrees_of_freedom
        X_transpose_X_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        std_err = np.sqrt(mse * X_transpose_X_inv[0, 0])
        t_stat = alpha / std_err
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_of_freedom))
    else:
        std_err = np.nan
        p_value = np.nan
    
    return alpha, betas, r_squared, p_value, std_err


def treynor_ratio(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray],
    risk_free_rate: Union[float, List[float], np.ndarray] = 0.0,
    annualization_factor: float = 1.0
) -> float:
    """
    Calculate the Treynor ratio, which measures excess return per unit of systematic risk.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    benchmark_returns : array-like
        Array of benchmark returns for the same periods
    risk_free_rate : float or array-like, default 0.0
        Risk-free rate for the same period as returns
    annualization_factor : float, default 1.0
        Factor to annualize the Treynor ratio
        
    Returns
    -------
    float
        The Treynor ratio
        
    Examples
    --------
    >>> treynor_ratio([0.01, 0.02, -0.01, 0.03, 0.01], [0.005, 0.01, -0.005, 0.02, 0.005], 0.001, 252)
    0.0378
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate = np.asarray(risk_free_rate)
        risk_free_rate = risk_free_rate.mean()
    
    excess_returns = returns - risk_free_rate
    excess_benchmark_returns = benchmark_returns - risk_free_rate
    
    beta = np.cov(excess_returns, excess_benchmark_returns)[0, 1] / np.var(excess_benchmark_returns)
    
    avg_excess_return = excess_returns.mean()
    
    treynor = avg_excess_return / beta
    
    if annualization_factor != 1.0:
        treynor = treynor * annualization_factor
        
    return treynor


def sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: Union[float, List[float], np.ndarray] = 0.0,
    target_return: float = 0.0,
    annualization_factor: float = 1.0
) -> float:
    """
    Calculate the Sortino ratio, which measures excess return per unit of downside risk.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    risk_free_rate : float or array-like, default 0.0
        Risk-free rate for the same period as returns
    target_return : float, default 0.0
        Minimum acceptable return
    annualization_factor : float, default 1.0
        Factor to annualize the Sortino ratio
        
    Returns
    -------
    float
        The Sortino ratio
        
    Examples
    --------
    >>> sortino_ratio([0.01, 0.02, -0.01, 0.03, 0.01], 0.001, 0.0, 252)
    3.7947331922020545
    """
    returns = np.asarray(returns)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate = np.asarray(risk_free_rate)
        risk_free_rate = risk_free_rate.mean()
    
    excess_returns = returns.mean() - risk_free_rate
    
    downside_returns = returns[returns < target_return] - target_return
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    sortino = excess_returns / downside_deviation
    
    if annualization_factor != 1.0:
        sortino = sortino * np.sqrt(annualization_factor)
        
    return sortino


def calmar_ratio(
    returns: Union[List[float], np.ndarray],
    max_drawdown: Optional[float] = None,
    annualization_factor: float = 1.0
) -> float:
    """
    Calculate the Calmar ratio, which measures return relative to maximum drawdown.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    max_drawdown : float, optional
        Maximum drawdown as a positive decimal. If None, it will be calculated from returns.
    annualization_factor : float, default 1.0
        Factor to annualize returns
        
    Returns
    -------
    float
        The Calmar ratio
        
    Examples
    --------
    >>> calmar_ratio([0.01, 0.02, -0.01, 0.03, 0.01], 0.15, 252)
    0.8
    """
    returns = np.asarray(returns)
    
    annualized_return = returns.mean() * annualization_factor
    
    if max_drawdown is None:
        cum_returns = (1 + returns).cumprod()
        
        running_max = np.maximum.accumulate(cum_returns)
        
        drawdowns = (cum_returns - running_max) / running_max
        
        max_drawdown = abs(drawdowns.min())
    
    max_drawdown = abs(max_drawdown)
    
    if max_drawdown == 0:
        return float('inf')
    
    calmar = annualized_return / max_drawdown
    
    return calmar


def omega_ratio(
    returns: Union[List[float], np.ndarray],
    threshold: float = 0.0,
    annualization_factor: float = 1.0
) -> float:
    """
    Calculate the Omega ratio, which measures the probability-weighted ratio of gains versus losses.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    threshold : float, default 0.0
        The threshold return
    annualization_factor : float, default 1.0
        Factor to annualize the threshold
        
    Returns
    -------
    float
        The Omega ratio
        
    Examples
    --------
    >>> omega_ratio([0.01, 0.02, -0.01, 0.03, 0.01], 0.005)
    2.0
    """
    returns = np.asarray(returns)
    
    if annualization_factor != 1.0:
        threshold = (1 + threshold) ** (1 / annualization_factor) - 1
    
    returns_above = returns[returns > threshold] - threshold
    returns_below = threshold - returns[returns < threshold]
    
    sum_gains = returns_above.sum() if len(returns_above) > 0 else 0
    sum_losses = returns_below.sum() if len(returns_below) > 0 else 0
    
    if sum_losses == 0:
        return float('inf')
    
    omega = sum_gains / sum_losses
    
    return omega


def kelly_criterion(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: Union[float, List[float], np.ndarray] = 0.0
) -> float:
    """
    Calculate the Kelly criterion, which maximizes the expected logarithmic growth rate.
    
    Parameters
    ----------
    returns : array-like
        Array of portfolio returns
    risk_free_rate : float or array-like, default 0.0
        Risk-free rate for the same period as returns
        
    Returns
    ------- 
    float
        The Kelly criterion
        
    Examples
    --------
    >>> kelly_criterion([0.01, 0.02, -0.01, 0.03, 0.01], 0.001)
    0.5
    """
    returns = np.asarray(returns)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate = np.asarray(risk_free_rate)
    
    excess_returns = returns - risk_free_rate
    
    mean_excess_returns = excess_returns.mean()
    
    std_excess_returns = excess_returns.std()
    
    kelly = mean_excess_returns / std_excess_returns
    
    return kelly
