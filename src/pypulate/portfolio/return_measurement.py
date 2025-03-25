"""
Return measurement functions for portfolio analysis.

This module provides functions for measuring returns in various portfolio scenarios,
including portfolios with no cash flows, portfolios with inflows and outflows,
and market-neutral and leveraged portfolios.

All functions support both Python lists and NumPy arrays as inputs.
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Sequence
from numpy.typing import ArrayLike, NDArray


def simple_return(
    end_value: Union[float, ArrayLike],
    start_value: Union[float, ArrayLike]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the simple return (percentage change) between two values.
    
    Parameters
    ----------
    end_value : float or array-like
        The ending value(s) of the investment
    start_value : float or array-like
        The starting value(s) of the investment
        
    Returns
    -------
    float or ndarray
        The simple return as a decimal (e.g., 0.05 for 5%)
        If array inputs are provided, returns an array of simple returns
        
    Examples
    --------
    >>> simple_return(105, 100)
    0.05
    >>> simple_return([105, 110, 108], [100, 100, 100])
    array([0.05, 0.1 , 0.08])
    >>> simple_return(np.array([105, 110]), np.array([100, 100]))
    array([0.05, 0.1 ])
    """
    if isinstance(end_value, (list, np.ndarray)) or isinstance(start_value, (list, np.ndarray)):
        end_value_arr = np.asarray(end_value, dtype=np.float64)
        start_value_arr = np.asarray(start_value, dtype=np.float64)
        return (end_value_arr - start_value_arr) / start_value_arr
    
    return (end_value - start_value) / start_value


def log_return(
    end_value: Union[float, ArrayLike],
    start_value: Union[float, ArrayLike]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the logarithmic (continuously compounded) return between two values.
    
    Parameters
    ----------
    end_value : float or array-like
        The ending value(s) of the investment
    start_value : float or array-like
        The starting value(s) of the investment
        
    Returns
    -------
    float or ndarray
        The logarithmic return
        If array inputs are provided, returns an array of logarithmic returns
        
    Examples
    --------
    >>> log_return(105, 100)
    0.04879016416929972
    >>> log_return([105, 110, 108], [100, 100, 100])
    array([0.04879016, 0.09531018, 0.07696104])
    >>> log_return(np.array([105, 110]), np.array([100, 100]))
    array([0.04879016, 0.09531018])
    """
    if isinstance(end_value, (list, np.ndarray)) or isinstance(start_value, (list, np.ndarray)):
        end_value_arr = np.asarray(end_value, dtype=np.float64)
        start_value_arr = np.asarray(start_value, dtype=np.float64)
        return np.log(end_value_arr / start_value_arr)
    
    return np.log(end_value / start_value)


def holding_period_return(
    prices: ArrayLike,
    dividends: Optional[ArrayLike] = None
) -> float:
    """
    Calculate the holding period return for a series of prices and optional dividends.
    
    Parameters
    ----------
    prices : array-like
        Array or list of prices over the holding period
    dividends : array-like, optional
        Array or list of dividends paid during the holding period
        
    Returns
    -------
    float
        The holding period return as a decimal
        
    Examples
    --------
    >>> holding_period_return([100, 102, 105, 103, 106])
    0.06
    >>> holding_period_return([100, 102, 105, 103, 106], [0, 1, 0, 2, 0])
    0.09
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    
    start_price = prices_arr[0]
    end_price = prices_arr[-1]
    
    if dividends is not None:
        dividends_arr = np.asarray(dividends, dtype=np.float64)
        total_dividends = np.sum(dividends_arr)
        return float((end_price - start_price + total_dividends) / start_price)
    else:
        return float((end_price - start_price) / start_price)


def annualized_return(
    total_return: Union[float, ArrayLike],
    years: Union[float, ArrayLike]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the annualized return from a total return over a period of years.
    
    Parameters
    ----------
    total_return : float or array-like
        The total return over the entire period as a decimal
    years : float or array-like
        The number of years in the period
        
    Returns
    -------
    float or ndarray
        The annualized return as a decimal
        If array inputs are provided, returns an array of annualized returns
        
    Examples
    --------
    >>> annualized_return(0.2, 2)
    0.09544511501033215
    >>> annualized_return([0.2, 0.3, 0.15], [2, 3, 1.5])
    [0.09544512, 0.09139288, 0.0976534 ]
    >>> annualized_return(np.array([0.4, 0.5]), 2)
    [0.18321596, 0.22474487]
    """
    if isinstance(total_return, (list, np.ndarray)) or isinstance(years, (list, np.ndarray)):
        total_return_arr = np.asarray(total_return, dtype=np.float64)
        years_arr = np.asarray(years, dtype=np.float64)
        return (1.0 + total_return_arr) ** (1.0 / years_arr) - 1.0
    
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def time_weighted_return(
    period_returns: ArrayLike
) -> float:
    """
    Calculate the time-weighted return from a series of period returns.
    
    Parameters
    ----------
    period_returns : array-like
        Array or list of returns for each period
        
    Returns
    -------
    float
        The time-weighted return as a decimal
        
    Examples
    --------
    >>> time_weighted_return([0.05, -0.02, 0.03, 0.04])
    0.10226479999999993
    """
    period_returns_arr = np.asarray(period_returns, dtype=np.float64)
    
    return float(np.prod(1.0 + period_returns_arr) - 1.0)


def money_weighted_return(
    cash_flows: ArrayLike,
    cash_flow_times: ArrayLike,
    final_value: float,
    initial_value: float = 0,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate the money-weighted return (internal rate of return) for a series of cash flows.
    
    Parameters
    ----------
    cash_flows : array-like
        Array or list of cash flows (positive for inflows, negative for outflows)
    cash_flow_times : array-like
        Array or list of times (in years) when each cash flow occurs
    final_value : float
        The final value of the investment
    initial_value : float, default 0
        The initial value of the investment
    max_iterations : int, default 100
        Maximum number of iterations for the numerical solver
    tolerance : float, default 1e-6
        Convergence tolerance for the numerical solver
        
    Returns
    -------
    float
        The money-weighted return (IRR) as a decimal
        
    Examples
    --------
    >>> money_weighted_return([-1000, -500, 1700], [0, 0.5, 1], 0)
    0.16120409753798307
    """
    cash_flows_arr = np.asarray(cash_flows, dtype=np.float64)
    cash_flow_times_arr = np.asarray(cash_flow_times, dtype=np.float64)
    
    # Special case for simple investment with initial value
    if len(cash_flows_arr) == 1 and initial_value > 0:
        return float((final_value / initial_value - 1) / cash_flow_times_arr[0])
        
    if initial_value != 0:
        cash_flows_arr = np.insert(cash_flows_arr, 0, -initial_value)  # Initial investment is a negative cash flow
        cash_flow_times_arr = np.insert(cash_flow_times_arr, 0, 0)
    
    cash_flows_arr = np.append(cash_flows_arr, final_value)
    
    # Ensure the last cash flow time is in the array
    if len(cash_flow_times_arr) > 0:
        cash_flow_times_arr = np.append(cash_flow_times_arr, cash_flow_times_arr[-1] + 1)
    else:
        cash_flow_times_arr = np.array([0, 1], dtype=np.float64)
    
    r = 0.1  
    
    for _ in range(max_iterations):
        f = np.sum(cash_flows_arr / (1.0 + r) ** cash_flow_times_arr)
        if abs(f) < tolerance:
            break
            
        df = np.sum(-cash_flow_times_arr * cash_flows_arr / (1.0 + r) ** (cash_flow_times_arr + 1.0))
        r_new = r - f / df
        
        if abs(r_new - r) < tolerance:
            r = r_new
            break
            
        r = r_new
    
    return r


# -------------------------------------------------------------------------
# No Cash Flows
# -------------------------------------------------------------------------

def arithmetic_return(
    prices: ArrayLike
) -> float:
    """
    Calculate the arithmetic average return from a series of prices.
    
    Parameters
    ----------
    prices : array-like
        Array or list of prices
        
    Returns
    -------
    float
        The arithmetic average return as a decimal
        
    Examples
    --------
    >>> arithmetic_return([100, 105, 103, 108, 110])
    0.024503647197821957
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    
    returns = np.diff(prices_arr) / prices_arr[:-1]
    return float(np.mean(returns))


def geometric_return(
    prices: ArrayLike
) -> float:
    """
    Calculate the geometric average return from a series of prices.
    
    Parameters
    ----------
    prices : array-like
        Array or list of prices
        
    Returns
    -------
    float
        The geometric average return as a decimal
        
    Examples
    --------
    >>> geometric_return([100, 105, 103, 108, 110])
    0.02411368908444511
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    
    returns = np.diff(prices_arr) / prices_arr[:-1]
    return float(np.prod(1.0 + returns) ** (1.0 / len(returns)) - 1.0)


def total_return_index(
    prices: ArrayLike,
    dividends: Optional[ArrayLike] = None
) -> NDArray[np.float64]:
    """
    Calculate the total return index from a series of prices and optional dividends.
    
    Parameters
    ----------
    prices : array-like
        Array or list of prices
    dividends : array-like, optional
        Array or list of dividends paid
        
    Returns
    -------
    numpy.ndarray
        The total return index
        
    Examples
    --------
    >>> total_return_index([100, 102, 105, 103, 106])
    [100., 102., 105., 103., 106.]
    >>> total_return_index([100, 102, 105, 103, 106], [0, 1, 0, 2, 0])
    [100.        , 103.        , 106.02941176, 106.02941176,
       109.11764706]
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    
    if dividends is None:
        dividends_arr = np.zeros_like(prices_arr, dtype=np.float64)
    else:
        dividends_arr = np.asarray(dividends, dtype=np.float64)
    
    if len(dividends_arr) != len(prices_arr):
        raise ValueError("Dividends array must be the same length as prices array")
    
    tri = np.zeros_like(prices_arr, dtype=np.float64)
    tri[0] = prices_arr[0]
    
    for i in range(1, len(prices_arr)):
        tri[i] = tri[i-1] * (prices_arr[i] + dividends_arr[i]) / prices_arr[i-1]
    
    return tri


# -------------------------------------------------------------------------
# Inflows and Outflows
# -------------------------------------------------------------------------

def dollar_weighted_return(
    cash_flows: ArrayLike,
    cash_flow_dates: ArrayLike,
    end_value: float
) -> float:
    """
    Calculate the dollar-weighted return (internal rate of return) for a series of cash flows.
    
    Parameters
    ----------
    cash_flows : array-like
        Array or list of cash flows (positive for inflows, negative for outflows)
    cash_flow_dates : array-like
        Array or list of dates (in days) when each cash flow occurs
    end_value : float
        The final value of the investment
        
    Returns
    -------
    float
        The dollar-weighted return as a decimal
        
    Examples
    --------
    >>> dollar_weighted_return([-1000, -500, 200], [0, 30, 60], 1400)
    0.36174448410245186
    """
    cash_flows_arr = np.asarray(cash_flows, dtype=np.float64)
    cash_flow_dates_arr = np.asarray(cash_flow_dates, dtype=np.float64)
    
    years_arr = (cash_flow_dates_arr - cash_flow_dates_arr[0]) / 365.0
    
    cash_flows_arr = np.append(cash_flows_arr, end_value)
    years_arr = np.append(years_arr, years_arr[-1] + (30.0 / 365.0)) 
    
    r = 0.1  
    max_iterations = 100
    tolerance = 1e-6
    
    for _ in range(max_iterations):
        f = np.sum(cash_flows_arr / (1.0 + r) ** years_arr)
        if abs(f) < tolerance:
            break
            
        df = np.sum(-years_arr * cash_flows_arr / (1.0 + r) ** (years_arr + 1.0))
        r_new = r - f / df
        
        if abs(r_new - r) < tolerance:
            r = r_new
            break
            
        r = r_new
    
    return r


def modified_dietz_return(
    start_value: float,
    end_value: float,
    cash_flows: ArrayLike,
    cash_flow_days: ArrayLike,
    total_days: int
) -> float:
    """
    Calculate the Modified Dietz return, which approximates the money-weighted return.
    
    Parameters
    ----------
    start_value : float
        The starting value of the investment
    end_value : float
        The ending value of the investment
    cash_flows : array-like
        Array or list of cash flows (positive for inflows, negative for outflows)
    cash_flow_days : array-like
        Array or list of days when each cash flow occurs (day 0 is the start)
    total_days : int
        Total number of days in the period
        
    Returns
    -------
    float
        The Modified Dietz return as a decimal
        
    Examples
    --------
    >>> modified_dietz_return(1000, 1200, [100, -50], [10, 20], 30)
    0.14285714285714285
    """
    cash_flows_arr = np.asarray(cash_flows, dtype=np.float64)
    cash_flow_days_arr = np.asarray(cash_flow_days, dtype=np.float64)
    
    weights = 1.0 - (cash_flow_days_arr / float(total_days))
    
    weighted_cash_flows = cash_flows_arr * weights
    
    return float((end_value - start_value - np.sum(cash_flows_arr)) / (start_value + np.sum(weighted_cash_flows)))


def linked_modified_dietz_return(
    period_returns: ArrayLike
) -> float:
    """
    Calculate the linked Modified Dietz return over multiple periods.
    
    Parameters
    ----------
    period_returns : array-like
        Array or list of Modified Dietz returns for each period
        
    Returns
    -------
    float
        The linked Modified Dietz return as a decimal
        
    Examples
    --------
    >>> linked_modified_dietz_return([0.05, -0.02, 0.03, 0.04])
    0.10226479999999993
    """
    period_returns_arr = np.asarray(period_returns, dtype=np.float64)
    
    return float(np.prod(1.0 + period_returns_arr) - 1.0)


# -------------------------------------------------------------------------
# Market-Neutral and Leveraged Portfolios
# -------------------------------------------------------------------------

def leveraged_return(
    unleveraged_return: Union[float, ArrayLike],
    leverage_ratio: Union[float, ArrayLike],
    borrowing_rate: Union[float, ArrayLike]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the return of a leveraged portfolio.
    
    Parameters
    ----------
    unleveraged_return : float or array-like
        The return of the unleveraged portfolio as a decimal
    leverage_ratio : float or array-like
        The leverage ratio (e.g., 2.0 for 2:1 leverage)
    borrowing_rate : float or array-like
        The borrowing rate as a decimal
        
    Returns
    -------
    float or ndarray
        The leveraged return as a decimal
        If array inputs are provided, returns an array of leveraged returns
        
    Examples
    --------
    >>> leveraged_return(0.10, 2.0, 0.05)
    0.15
    >>> leveraged_return([0.10, 0.15], [2.0, 1.5], 0.05)
    [0.15, 0.2 ]
    >>> leveraged_return(0.10, [2.0, 3.0], [0.05, 0.06])
    [0.15, 0.18]
    """
    if (isinstance(unleveraged_return, (list, np.ndarray)) or 
        isinstance(leverage_ratio, (list, np.ndarray)) or 
        isinstance(borrowing_rate, (list, np.ndarray))):
        unleveraged_return_arr = np.asarray(unleveraged_return, dtype=np.float64)
        leverage_ratio_arr = np.asarray(leverage_ratio, dtype=np.float64)
        borrowing_rate_arr = np.asarray(borrowing_rate, dtype=np.float64)
        return unleveraged_return_arr * leverage_ratio_arr - borrowing_rate_arr * (leverage_ratio_arr - 1.0)
    
    return unleveraged_return * leverage_ratio - borrowing_rate * (leverage_ratio - 1.0)


def market_neutral_return(
    long_return: Union[float, ArrayLike],
    short_return: Union[float, ArrayLike],
    long_weight: Union[float, ArrayLike] = 0.5,
    short_weight: Union[float, ArrayLike] = 0.5,
    short_borrowing_cost: Union[float, ArrayLike] = 0.0
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the return of a market-neutral portfolio with long and short positions.
    
    Parameters
    ----------
    long_return : float or array-like
        The return of the long portfolio as a decimal
    short_return : float or array-like
        The return of the short portfolio as a decimal
    long_weight : float or array-like, default 0.5
        The weight of the long portfolio
    short_weight : float or array-like, default 0.5
        The weight of the short portfolio
    short_borrowing_cost : float or array-like, default 0.0
        The cost of borrowing for the short position as a decimal
        
    Returns
    -------
    float or ndarray
        The market-neutral return as a decimal
        If array inputs are provided, returns an array of market-neutral returns
        
    Examples
    --------
    >>> market_neutral_return(0.08, -0.05, 0.6, 0.4, 0.01)
    0.064
    >>> market_neutral_return([0.08, 0.10], [-0.05, -0.03], 0.6, 0.4, 0.01)
    [0.064, 0.068]
    >>> market_neutral_return(0.08, -0.05, [0.6, 0.7], [0.4, 0.3], [0.01, 0.02])
    [0.064, 0.065]
    """
    if (isinstance(long_return, (list, np.ndarray)) or 
        isinstance(short_return, (list, np.ndarray)) or
        isinstance(long_weight, (list, np.ndarray)) or
        isinstance(short_weight, (list, np.ndarray)) or
        isinstance(short_borrowing_cost, (list, np.ndarray))):
        long_return_arr = np.asarray(long_return, dtype=np.float64)
        short_return_arr = np.asarray(short_return, dtype=np.float64)
        long_weight_arr = np.asarray(long_weight, dtype=np.float64)
        short_weight_arr = np.asarray(short_weight, dtype=np.float64)
        short_borrowing_cost_arr = np.asarray(short_borrowing_cost, dtype=np.float64)
        
        long_contribution = long_return_arr * long_weight_arr
        short_contribution = -short_return_arr * short_weight_arr
        short_cost = short_borrowing_cost_arr * short_weight_arr
        
        return long_contribution + short_contribution - short_cost
    
    long_contribution = long_return * long_weight
    short_contribution = -short_return * short_weight
    short_cost = short_borrowing_cost * short_weight
    
    return long_contribution + short_contribution - short_cost


def beta_adjusted_return(
    portfolio_return: Union[float, ArrayLike],
    benchmark_return: Union[float, ArrayLike],
    portfolio_beta: Union[float, ArrayLike]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the beta-adjusted return (alpha) of a portfolio.
    
    Parameters
    ----------
    portfolio_return : float or array-like
        The return of the portfolio as a decimal
    benchmark_return : float or array-like
        The return of the benchmark as a decimal
    portfolio_beta : float or array-like
        The beta of the portfolio relative to the benchmark
        
    Returns
    -------
    float or ndarray
        The beta-adjusted return (alpha) as a decimal
        If array inputs are provided, returns an array of beta-adjusted returns
        
    Examples
    --------
    >>> beta_adjusted_return(0.12, 0.10, 1.2)
    0.0
    >>> beta_adjusted_return([0.12, 0.15], [0.10, 0.08], 1.2)
    [0.   , 0.054]
    >>> beta_adjusted_return(0.12, 0.10, [1.2, 1.5])
    [ 0.  , -0.03]
    """
    if (isinstance(portfolio_return, (list, np.ndarray)) or 
        isinstance(benchmark_return, (list, np.ndarray)) or 
        isinstance(portfolio_beta, (list, np.ndarray))):
        portfolio_return_arr = np.asarray(portfolio_return, dtype=np.float64)
        benchmark_return_arr = np.asarray(benchmark_return, dtype=np.float64)
        portfolio_beta_arr = np.asarray(portfolio_beta, dtype=np.float64)
        
        expected_return = portfolio_beta_arr * benchmark_return_arr
        return portfolio_return_arr - expected_return
    
    expected_return = portfolio_beta * benchmark_return
    return portfolio_return - expected_return


def long_short_equity_return(
    long_portfolio_return: Union[float, ArrayLike],
    short_portfolio_return: Union[float, ArrayLike],
    long_exposure: Union[float, ArrayLike],
    short_exposure: Union[float, ArrayLike],
    risk_free_rate: Union[float, ArrayLike] = 0.0,
    short_rebate: Union[float, ArrayLike] = 0.0
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the return of a long-short equity portfolio.
    
    Parameters
    ----------
    long_portfolio_return : float or array-like
        The return of the long portfolio as a decimal
    short_portfolio_return : float or array-like
        The return of the short portfolio as a decimal
    long_exposure : float or array-like
        The exposure of the long portfolio as a decimal of NAV
    short_exposure : float or array-like
        The exposure of the short portfolio as a decimal of NAV
    risk_free_rate : float or array-like, default 0.0
        The risk-free rate as a decimal
    short_rebate : float or array-like, default 0.0
        The rebate received on short proceeds as a decimal
        
    Returns
    -------
    float or ndarray
        The return of the long-short equity portfolio as a decimal
        If array inputs are provided, returns an array of long-short equity returns
        
    Examples
    --------
    >>> long_short_equity_return(0.10, -0.05, 1.0, 0.5, 0.02, 0.01)
    0.14
    >>> long_short_equity_return([0.10, 0.12], [-0.05, -0.03], 1.0, 0.5, 0.02, 0.01)
    array([0.14, 0.15])
    >>> long_short_equity_return(0.10, -0.05, [1.0, 0.8], [0.5, 0.4], [0.02, 0.03], 0.01)
    array([0.14 , 0.122])
    """
    if (isinstance(long_portfolio_return, (list, np.ndarray)) or 
        isinstance(short_portfolio_return, (list, np.ndarray)) or
        isinstance(long_exposure, (list, np.ndarray)) or
        isinstance(short_exposure, (list, np.ndarray)) or
        isinstance(risk_free_rate, (list, np.ndarray)) or
        isinstance(short_rebate, (list, np.ndarray))):
        long_portfolio_return_arr = np.asarray(long_portfolio_return, dtype=np.float64)
        short_portfolio_return_arr = np.asarray(short_portfolio_return, dtype=np.float64)
        long_exposure_arr = np.asarray(long_exposure, dtype=np.float64)
        short_exposure_arr = np.asarray(short_exposure, dtype=np.float64)
        risk_free_rate_arr = np.asarray(risk_free_rate, dtype=np.float64)
        short_rebate_arr = np.asarray(short_rebate, dtype=np.float64)
        
        cash_exposure_arr = 1.0 - long_exposure_arr + short_exposure_arr
        
        long_contribution = long_portfolio_return_arr * long_exposure_arr
        short_contribution = -short_portfolio_return_arr * short_exposure_arr
        cash_contribution = risk_free_rate_arr * cash_exposure_arr
        rebate_contribution = short_rebate_arr * short_exposure_arr
        
        return long_contribution + short_contribution + cash_contribution + rebate_contribution
    
    cash_exposure = 1.0 - long_exposure + short_exposure
    
    long_contribution = long_portfolio_return * long_exposure
    short_contribution = -short_portfolio_return * short_exposure
    cash_contribution = risk_free_rate * cash_exposure
    rebate_contribution = short_rebate * short_exposure
    
    return long_contribution + short_contribution + cash_contribution + rebate_contribution
