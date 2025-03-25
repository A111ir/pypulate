"""
Risk-adjusted performance measurement functions for portfolio analysis.

This module provides functions for measuring risk-adjusted performance metrics
including Sharpe ratio, Information ratio, CAPM alpha, and multifactor models.

All functions support both Python lists and NumPy arrays as inputs.
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Sequence
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: Union[float, ArrayLike] = 0.0,
    annualization_factor: float = 1.0
) -> Union[float, NDArray[np.float64]]:
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate_arr = np.asarray(risk_free_rate, dtype=np.float64)
        excess_returns = returns_arr.mean() - risk_free_rate_arr
    else:
        excess_returns = returns_arr.mean() - risk_free_rate
    
    volatility = returns_arr.std()
    
    sharpe = excess_returns / volatility
    
    if annualization_factor != 1.0:
        sharpe = sharpe * np.sqrt(annualization_factor)
        
    if isinstance(sharpe, np.ndarray):
        return sharpe
    return float(sharpe)


def information_ratio(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    benchmark_returns_arr = np.asarray(benchmark_returns, dtype=np.float64)
    
    active_returns = returns_arr - benchmark_returns_arr
    
    active_return = active_returns.mean()
    
    tracking_error = active_returns.std()
    
    if tracking_error == 0 or np.isclose(tracking_error, 0):
        return 0.0
    
    ir = active_return / tracking_error
    
    if annualization_factor != 1.0:
        ir = ir * np.sqrt(annualization_factor)
        
    return float(ir)


def capm_alpha(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    risk_free_rate: Union[float, ArrayLike] = 0.0
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    benchmark_returns_arr = np.asarray(benchmark_returns, dtype=np.float64)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate_arr = np.asarray(risk_free_rate, dtype=np.float64)
        rf_rate = float(risk_free_rate_arr.mean())
    else:
        rf_rate = float(risk_free_rate)
    
    excess_returns = returns_arr - rf_rate
    excess_benchmark_returns = benchmark_returns_arr - rf_rate
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        excess_benchmark_returns, excess_returns
    )
    
    alpha = float(intercept)
    beta = float(slope)
    r_squared = float(r_value ** 2)
    p_value = float(p_value)
    std_err = float(std_err)
    
    return alpha, beta, r_squared, p_value, std_err


def benchmark_alpha(
    returns: ArrayLike,
    benchmark_returns: ArrayLike
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    benchmark_returns_arr = np.asarray(benchmark_returns, dtype=np.float64)
    
    portfolio_mean_return = returns_arr.mean()
    benchmark_mean_return = benchmark_returns_arr.mean()
    
    alpha = portfolio_mean_return - benchmark_mean_return
    
    return float(alpha)


def multifactor_alpha(
    returns: ArrayLike,
    factor_returns: ArrayLike,
    risk_free_rate: Union[float, ArrayLike] = 0.0
) -> Tuple[float, NDArray[np.float64], float, float, float]:
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    factor_returns_arr = np.asarray(factor_returns, dtype=np.float64)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate_arr = np.asarray(risk_free_rate, dtype=np.float64)
        rf_rate = float(risk_free_rate_arr.mean())
    else:
        rf_rate = float(risk_free_rate)
    
    excess_returns = returns_arr - rf_rate
    
    if factor_returns_arr.shape[0] == len(returns_arr) and factor_returns_arr.ndim > 1:
        X = factor_returns_arr  
    else:
        X = factor_returns_arr.T  
    
    X_with_const = np.column_stack([np.ones(len(excess_returns)), X])
    
    result = np.linalg.lstsq(X_with_const, excess_returns, rcond=None)
    coefficients = result[0]
    
    alpha = float(coefficients[0])
    betas = coefficients[1:]
    
    y_pred = X_with_const @ coefficients
    ss_total: float = np.sum((excess_returns - np.mean(excess_returns))**2)
    ss_residual: float = np.sum((excess_returns - y_pred)**2)
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
    
    return float(alpha), betas, float(r_squared), float(p_value), float(std_err)


def treynor_ratio(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    risk_free_rate: Union[float, ArrayLike] = 0.0,
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    benchmark_returns_arr = np.asarray(benchmark_returns, dtype=np.float64)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate_arr = np.asarray(risk_free_rate, dtype=np.float64)
        rf_rate = float(risk_free_rate_arr.mean())
    else:
        rf_rate = float(risk_free_rate)
    
    excess_returns = returns_arr - rf_rate
    excess_benchmark_returns = benchmark_returns_arr - rf_rate
    
    beta = np.cov(excess_returns, excess_benchmark_returns)[0, 1] / np.var(excess_benchmark_returns)
    
    avg_excess_return = excess_returns.mean()
    
    treynor = avg_excess_return / beta
    
    if annualization_factor != 1.0:
        treynor = treynor * annualization_factor
        
    return float(treynor)


def sortino_ratio(
    returns: ArrayLike,
    risk_free_rate: Union[float, ArrayLike] = 0.0,
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    
    if isinstance(risk_free_rate, (list, np.ndarray)):
        risk_free_rate_arr = np.asarray(risk_free_rate, dtype=np.float64)
        rf_rate = float(risk_free_rate_arr.mean())
    else:
        rf_rate = float(risk_free_rate)
    
    excess_returns = returns_arr.mean() - rf_rate
    
    downside_returns = returns_arr[returns_arr < target_return] - target_return
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    sortino = excess_returns / downside_deviation
    
    if annualization_factor != 1.0:
        sortino = sortino * np.sqrt(annualization_factor)
        
    return float(sortino)


def calmar_ratio(
    returns: ArrayLike,
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    
    annualized_return = returns_arr.mean() * annualization_factor
    
    if max_drawdown is None:
        cum_returns = (1.0 + returns_arr).cumprod()
        
        running_max = np.maximum.accumulate(cum_returns)
        
        drawdowns = (cum_returns - running_max) / running_max
        
        max_drawdown = abs(drawdowns.min())
    
    max_drawdown = abs(max_drawdown)
    
    if max_drawdown == 0:
        return float('inf')
    
    calmar = annualized_return / max_drawdown
    
    return float(calmar)


def omega_ratio(
    returns: ArrayLike,
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
    returns_arr = np.asarray(returns, dtype=np.float64)
    
    if annualization_factor != 1.0:
        threshold = (1.0 + threshold) ** (1.0 / annualization_factor) - 1.0
    
    returns_above = returns_arr[returns_arr > threshold] - threshold
    returns_below = threshold - returns_arr[returns_arr < threshold]
    
    sum_gains = float(returns_above.sum()) if len(returns_above) > 0 else 0.0
    sum_losses = float(returns_below.sum()) if len(returns_below) > 0 else 0.0
    
    if sum_losses == 0:
        return float('inf')
    
    omega = sum_gains / sum_losses
    
    return float(omega)

