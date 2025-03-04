"""
Risk measurement functions for portfolio analysis.

This module provides various risk metrics used in portfolio management and financial analysis.
"""

import numpy as np
from typing import Union, Tuple, Optional, List, Any, cast, TypeVar
from scipy import stats



def standard_deviation(returns: Union[List[float], np.ndarray], annualize: bool = False, periods_per_year: int = 252) -> float:
    """
    Calculate the standard deviation of returns.
    
    Parameters
    ----------
    returns : list or np.ndarray
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
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
        
    std = np.std(returns, ddof=1)  
    
    if annualize:
        std = std * np.sqrt(periods_per_year)
        
    return std


def semi_standard_deviation(
        returns: Union[List[float], np.ndarray], 
        threshold: float = 0.0, 
        annualize: bool = False, 
        periods_per_year: int = 252) -> float:
    """
    Calculate the semi-standard deviation of returns below a threshold.
    
    Parameters
    ----------
    returns : list or np.ndarray
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
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
        
    downside_returns = returns[returns < threshold]
    
    if len(downside_returns) == 0:
        return 0.0
    
    semi_std = np.std(downside_returns, ddof=1)
    
    if annualize:
        semi_std = semi_std * np.sqrt(periods_per_year)
        
    return semi_std


def tracking_error(portfolio_returns: Union[List[float], np.ndarray], 
                  benchmark_returns: Union[List[float], np.ndarray], 
                  annualize: bool = False, periods_per_year: int = 252) -> float:
    """
    Calculate the tracking error between portfolio returns and benchmark returns.
    
    Parameters
    ----------
    portfolio_returns : list or np.ndarray
        Array or list of portfolio returns
    benchmark_returns : list or np.ndarray
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
    if not isinstance(portfolio_returns, np.ndarray):
        portfolio_returns = np.array(portfolio_returns)
    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)
    
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio returns and benchmark returns must have the same length")
    
    excess_returns = portfolio_returns - benchmark_returns
    
    te = np.std(excess_returns, ddof=1)
    
    if annualize:
        te = te * np.sqrt(periods_per_year)
        
    return te


def capm_beta(portfolio_returns: Union[List[float], np.ndarray], 
             market_returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the CAPM beta of a portfolio.
    
    Parameters
    ----------
    portfolio_returns : list or np.ndarray
        Array or list of portfolio returns
    market_returns : list or np.ndarray
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
    if not isinstance(portfolio_returns, np.ndarray):
        portfolio_returns = np.array(portfolio_returns)
    if not isinstance(market_returns, np.ndarray):
        market_returns = np.array(market_returns)
    
    if len(portfolio_returns) != len(market_returns):
        raise ValueError("Portfolio returns and market returns must have the same length")
    
    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns, ddof=1)
    
    beta = covariance / market_variance
    
    return beta


def value_at_risk(returns: Union[List[float], np.ndarray], confidence_level: float = 0.95, 
                 method: str = 'historical', parametric_mean: Optional[float] = None,
                 parametric_std: Optional[float] = None, 
                 current_value: float = 1.0) -> float:
    """
    Calculate the Value-at-Risk (VaR) of a portfolio.
    
    Parameters
    ----------
    returns : list or np.ndarray
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
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
        
    if method == 'historical':
        var_percentile = 1 - confidence_level
        var_return = np.percentile(returns, var_percentile * 100)
        var_value = current_value * -var_return  
        
    elif method == 'parametric':
        if parametric_mean is None:
            parametric_mean = np.mean(returns)
        if parametric_std is None:
            parametric_std = np.std(returns, ddof=1)
            
        z_score = stats.norm.ppf(1 - confidence_level)
        var_return = parametric_mean + z_score * parametric_std
        var_value = current_value * -var_return  
        
    elif method == 'monte_carlo':
        if parametric_mean is None:
            parametric_mean = np.mean(returns)
        if parametric_std is None:
            parametric_std = np.std(returns, ddof=1)
            
        np.random.seed(42) 
        simulated_returns = np.random.normal(parametric_mean, parametric_std, 10000)
        
        var_percentile = 1 - confidence_level
        var_return = np.percentile(simulated_returns, var_percentile * 100)
        var_value = current_value * -var_return 
        
    else:
        raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
    
    return max(0, var_value) 



def covariance_matrix(returns_matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Calculate the covariance matrix of returns.
    
    Parameters
    ----------
    returns_matrix : list of lists or np.ndarray
        Matrix of returns where each column represents an asset
        
    Returns
    -------
    np.ndarray or list of lists
        Covariance matrix
        
    Notes
    -----
    The covariance matrix measures how returns of different assets move together.
    """
    if not isinstance(returns_matrix, np.ndarray):
        returns_matrix = np.array(returns_matrix)
        
    cov_matrix = np.cov(returns_matrix, rowvar=False, ddof=1)
    
    return cov_matrix



def correlation_matrix(returns_matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Calculate the correlation matrix of returns.
    
    Parameters
    ----------
    returns_matrix : list of lists or np.ndarray
        Matrix of returns where each column represents an asset

    Returns
    -------
    np.ndarray or list of lists
        Correlation matrix
        
    Notes
    -----
    The correlation matrix measures the strength of the relationship between
    returns of different assets, normalized to be between -1 and 1.
    """
    if not isinstance(returns_matrix, np.ndarray):
        returns_matrix = np.array(returns_matrix)
        
    corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
    
    return corr_matrix


def conditional_value_at_risk(returns: Union[List[float], np.ndarray], confidence_level: float = 0.95,
                             method: str = 'historical', current_value: float = 1.0) -> float:
    """
    Calculate the Conditional Value-at-Risk (CVaR) of a portfolio.
    
    Parameters
    ----------
    returns : list or np.ndarray
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
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
        
    if method == 'historical':
        var_percentile = 1 - confidence_level
        var_threshold = np.percentile(returns, var_percentile * 100)
        
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        cvar_return = np.mean(tail_returns)
        cvar_value = current_value * -cvar_return
        
    elif method == 'parametric':
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        
        z_score = stats.norm.ppf(1 - confidence_level)
        pdf_z = stats.norm.pdf(z_score)
        cdf_z = 1 - confidence_level
        
        cvar_return = mean - std * pdf_z / cdf_z
        cvar_value = current_value * -cvar_return 
        
    else:
        raise ValueError("Method must be 'historical' or 'parametric'")
    
    return max(0, cvar_value)



def drawdown(returns: Union[List[float], np.ndarray], as_list: bool = False) -> Union[Tuple[np.ndarray, float, int, int], Tuple[List[float], float, int, int]]:
    """
    Calculate the drawdown, maximum drawdown, and drawdown duration of returns.
    
    Parameters
    ----------
    returns : list or np.ndarray
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
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
        
    cum_returns = (1 + returns).cumprod()
    
    running_max = np.maximum.accumulate(cum_returns)
    
    drawdowns = (cum_returns - running_max) / running_max
    
    max_drawdown = np.min(drawdowns)
    end_idx = int(np.argmin(drawdowns))  
    
    start_idx = int(np.argmax(cum_returns[:end_idx])) if end_idx > 0 else 0  
    
    if as_list:
        return cast(List[float], drawdowns.tolist()), float(-max_drawdown), start_idx, end_idx
    return drawdowns, float(-max_drawdown), start_idx, end_idx
