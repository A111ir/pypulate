"""
Risk measurement functions for portfolio analysis.

This module provides various risk metrics used in portfolio management and financial analysis.
"""

import numpy as np
from typing import Union, Tuple, Optional, List
from numpy.typing import ArrayLike, NDArray
from scipy import stats



def standard_deviation(returns: ArrayLike, annualize: bool = False, periods_per_year: int = 252) -> float:
    """
    Calculate the standard deviation of returns.
    
    Parameters
    ----------
    returns : array-like
        Array or list of returns
    annualize : bool, default False
        Whether to annualize the standard deviation
    periods_per_year : int, default 252
        Number of periods in a year (252 for daily returns, 12 for monthly, 4 for quarterly)
        
    Returns
    -------
    float
        Standard deviation of returns
        
    Notes
    -----
    Standard deviation measures the dispersion of returns around the mean.
    It is the square root of the variance.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
        
    std = np.std(returns_arr, ddof=1)  
    
    if annualize:
        std = std * np.sqrt(periods_per_year)
        
    return float(std)


def semi_standard_deviation(
        returns: ArrayLike, 
        threshold: float = 0.0, 
        annualize: bool = False, 
        periods_per_year: int = 252) -> float:
    """
    Calculate the semi-standard deviation of returns below a threshold.
    
    Parameters
    ----------
    returns : array-like
        Array or list of returns
    threshold : float, default 0.0
        Threshold below which to calculate semi-standard deviation
    annualize : bool, default False
        Whether to annualize the semi-standard deviation
    periods_per_year : int, default 252
        Number of periods in a year (252 for daily returns, 12 for monthly, 4 for quarterly)
        
    Returns
    -------
    float
        Semi-standard deviation of returns
        
    Notes
    -----
    Semi-standard deviation only considers returns below the threshold (typically 0),
    making it a measure of downside risk.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
        
    downside_returns = returns_arr[returns_arr < threshold]
    
    if len(downside_returns) == 0:
        return 0.0
    
    semi_std = np.std(downside_returns, ddof=1)
    
    if annualize:
        semi_std = semi_std * np.sqrt(periods_per_year)
        
    return float(semi_std)


def tracking_error(portfolio_returns: ArrayLike, 
                  benchmark_returns: ArrayLike, 
                  annualize: bool = False, periods_per_year: int = 252) -> float:
    """
    Calculate the tracking error between portfolio returns and benchmark returns.
    
    Parameters
    ----------
    portfolio_returns : array-like
        Array or list of portfolio returns
    benchmark_returns : array-like
        Array or list of benchmark returns
    annualize : bool, default False
        Whether to annualize the tracking error
    periods_per_year : int, default 252
        Number of periods in a year (252 for daily returns, 12 for monthly, 4 for quarterly)
        
    Returns
    -------
    float
        Tracking error
        
    Notes
    -----
    Tracking error measures how closely a portfolio follows its benchmark.
    It is the standard deviation of the difference between portfolio and benchmark returns.
    """
    portfolio_returns_arr = np.asarray(portfolio_returns, dtype=np.float64)
    benchmark_returns_arr = np.asarray(benchmark_returns, dtype=np.float64)
    
    if len(portfolio_returns_arr) != len(benchmark_returns_arr):
        raise ValueError("Portfolio returns and benchmark returns must have the same length")
    
    excess_returns = portfolio_returns_arr - benchmark_returns_arr
    
    te = np.std(excess_returns, ddof=1)
    
    if annualize:
        te = te * np.sqrt(periods_per_year)
        
    return float(te)


def capm_beta(portfolio_returns: ArrayLike, 
             market_returns: ArrayLike) -> float:
    """
    Calculate the CAPM beta of a portfolio.
    
    Parameters
    ----------
    portfolio_returns : array-like
        Array or list of portfolio returns
    market_returns : array-like
        Array or list of market returns
        
    Returns
    -------
    float
        CAPM beta
        
    Notes
    -----
    Beta measures the sensitivity of portfolio returns to market returns.
    It is the covariance of portfolio returns and market returns divided by the variance of market returns.
    """
    portfolio_returns_arr = np.asarray(portfolio_returns, dtype=np.float64)
    market_returns_arr = np.asarray(market_returns, dtype=np.float64)
    
    if len(portfolio_returns_arr) != len(market_returns_arr):
        raise ValueError("Portfolio returns and market returns must have the same length")
    
    covariance = np.cov(portfolio_returns_arr, market_returns_arr)[0, 1]
    market_variance = np.var(market_returns_arr, ddof=1)
    
    beta = covariance / market_variance
    
    return float(beta)


def value_at_risk(returns: ArrayLike, confidence_level: float = 0.95, 
                 method: str = 'historical', parametric_mean: Optional[float] = None,
                 parametric_std: Optional[float] = None, 
                 current_value: float = 1.0) -> float:
    """
    Calculate the Value-at-Risk (VaR) of a portfolio.
    
    Parameters
    ----------
    returns : array-like
        Array or list of returns
    confidence_level : float, default 0.95
        Confidence level for VaR calculation (e.g., 0.95 for 95% confidence)
    method : str, default 'historical'
        Method for calculating VaR ('historical', 'parametric', or 'monte_carlo')
    parametric_mean : float, optional
        Mean for parametric VaR calculation (if None, calculated from returns)
    parametric_std : float, optional
        Standard deviation for parametric VaR calculation (if None, calculated from returns)
    current_value : float, default 1.0
        Current value of the portfolio
        
    Returns
    -------
    float
        Value-at-Risk (VaR) as a positive number representing the potential loss
        
    Notes
    -----
    VaR measures the potential loss in value of a portfolio over a defined period
    for a given confidence interval.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    var_value: float = 0.0
        
    if method == 'historical':
        var_percentile = 1 - confidence_level
        var_return = np.percentile(returns_arr, var_percentile * 100)
        var_value = current_value * -float(var_return)
        
    elif method == 'parametric':
        param_mean: float = float(np.mean(returns_arr)) if parametric_mean is None else parametric_mean
        param_std: float = float(np.std(returns_arr, ddof=1)) if parametric_std is None else parametric_std
            
        z_score = stats.norm.ppf(1 - confidence_level)
        var_return = param_mean + z_score * param_std
        var_value = current_value * -float(var_return)
        
    elif method == 'monte_carlo':
        mc_mean: float = float(np.mean(returns_arr)) if parametric_mean is None else parametric_mean
        mc_std: float = float(np.std(returns_arr, ddof=1)) if parametric_std is None else parametric_std
            
        np.random.seed(42) 
        simulated_returns = np.random.normal(mc_mean, mc_std, 10000)
        
        var_percentile = 1 - confidence_level
        var_return = np.percentile(simulated_returns, var_percentile * 100)
        var_value = current_value * -float(var_return)
        
    else:
        raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
    
    return float(max(0.0, var_value))


def covariance_matrix(returns_matrix: ArrayLike) -> NDArray[np.float64]:
    """
    Calculate the covariance matrix of returns.
    
    Parameters
    ----------
    returns_matrix : array-like
        Matrix of returns where each column represents an asset
        
    Returns
    -------
    ndarray
        Covariance matrix
        
    Notes
    -----
    The covariance matrix measures how returns of different assets move together.
    """
    returns_matrix_arr = np.asarray(returns_matrix, dtype=np.float64)
        
    cov_matrix = np.cov(returns_matrix_arr, rowvar=False, ddof=1)
    
    return cov_matrix


def correlation_matrix(returns_matrix: ArrayLike) -> NDArray[np.float64]:
    """
    Calculate the correlation matrix of returns.
    
    Parameters
    ----------
    returns_matrix : array-like
        Matrix of returns where each column represents an asset

    Returns
    -------
    ndarray
        Correlation matrix
        
    Notes
    -----
    The correlation matrix measures the strength of the relationship between
    returns of different assets, normalized to be between -1 and 1.
    """
    returns_matrix_arr = np.asarray(returns_matrix, dtype=np.float64)
        
    corr_matrix = np.corrcoef(returns_matrix_arr, rowvar=False)
    
    return corr_matrix


def conditional_value_at_risk(
    returns: ArrayLike,
    confidence_level: float = 0.95,
    method: str = 'historical',
    current_value: float = 1.0
) -> float:
    """
    Calculate the Conditional Value-at-Risk (CVaR) of a portfolio.
    
    Parameters
    ----------
    returns : array-like
        Array or list of returns
    confidence_level : float, default 0.95
        Confidence level for CVaR calculation (e.g., 0.95 for 95% confidence)
    method : str, default 'historical'
        Method for calculating CVaR ('historical' or 'parametric')
    current_value : float, default 1.0
        Current value of the portfolio
        
    Returns
    -------
    float
        Conditional Value-at-Risk (CVaR) as a positive number representing the potential loss
        
    Notes
    -----
    CVaR, also known as Expected Shortfall, measures the expected loss given that
    the loss exceeds the VaR threshold. It provides a more conservative risk measure than VaR.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    cvar_value = 0.0
        
    if method == 'historical':
        var_percentile = 1 - confidence_level
        var_threshold = np.percentile(returns_arr, var_percentile * 100)
        
        tail_returns = returns_arr[returns_arr <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        cvar_return = np.mean(tail_returns)
        cvar_value = current_value * -float(cvar_return)
        
    elif method == 'parametric':
        mean = float(np.mean(returns_arr))
        std = float(np.std(returns_arr, ddof=1))
        
        z_score = stats.norm.ppf(1 - confidence_level)
        pdf_z = float(stats.norm.pdf(z_score))
        cdf_z = 1 - confidence_level
        
        # Restructured calculation to avoid typing issues
        calc_result = mean - (std * pdf_z / cdf_z)
        # Explicitly convert to float and calculate final value
        cvar_value = current_value * -float(calc_result)
        
    else:
        raise ValueError("Method must be 'historical' or 'parametric'")
    
    return float(max(0.0, cvar_value))


def drawdown(returns: ArrayLike, as_list: bool = False) -> Tuple[Union[NDArray[np.float64], List[float]], float, int, int]:
    """
    Calculate the drawdown, maximum drawdown, and drawdown duration of returns.
    
    Parameters
    ----------
    returns : array-like
        Array or list of returns
    as_list : bool, default False
        If True, returns the drawdowns as a list instead of numpy array
        
    Returns
    -------
    Tuple containing:
        - Array or list of drawdowns
        - Maximum drawdown (as a positive number)
        - Start index of maximum drawdown
        - End index of maximum drawdown
        
    Notes
    -----
    Drawdown measures the decline from a historical peak in cumulative returns.
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
        
    cum_returns = (1 + returns_arr).cumprod()
    
    running_max = np.maximum.accumulate(cum_returns)
    
    drawdowns = (cum_returns - running_max) / running_max
    
    min_drawdown: float = float(np.min(drawdowns))
    end_idx = int(np.argmin(drawdowns))  
    
    start_idx = int(np.argmax(cum_returns[:end_idx])) if end_idx > 0 else 0  
    
    if as_list:
        return drawdowns.tolist(), float(-min_drawdown), start_idx, end_idx
    return drawdowns, float(-min_drawdown), start_idx, end_idx
