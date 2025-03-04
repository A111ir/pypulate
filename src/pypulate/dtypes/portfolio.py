"""
Portfolio Module

This module provides a class for calculating various portfolio metrics
including returns, risk-adjusted performance, and risk measurements.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
from ..portfolio.return_measurement import (
    simple_return, log_return, holding_period_return,
    annualized_return, time_weighted_return, money_weighted_return,
    arithmetic_return, geometric_return, total_return_index,
    dollar_weighted_return, modified_dietz_return, linked_modified_dietz_return,
    leveraged_return, market_neutral_return, beta_adjusted_return,
    long_short_equity_return
)
from ..portfolio.risk_adjusted import (
    sharpe_ratio, information_ratio, capm_alpha,
    benchmark_alpha, multifactor_alpha, treynor_ratio,
    sortino_ratio, calmar_ratio, omega_ratio, kelly_criterion
)
from ..portfolio.risk_measurement import (
    standard_deviation, semi_standard_deviation, tracking_error,
    capm_beta, value_at_risk, covariance_matrix, correlation_matrix,
    conditional_value_at_risk, drawdown
)


class Portfolio:
    """
    A class for calculating various portfolio metrics and assessing portfolio health.
    
    This class provides methods for calculating portfolio returns, risk-adjusted
    performance metrics, and risk measurements, while maintaining state to assess
    overall portfolio health.
    
    Examples
    --------
    >>> from pypulate.dtypes import Portfolio
    >>> portfolio = Portfolio()
    >>> returns = portfolio.simple_return(105, 100)
    >>> sharpe = portfolio.sharpe_ratio([0.01, 0.02, -0.01, 0.03, 0.01])
    >>> health = portfolio.health
    """
    
    def __init__(self):
        """Initialize the Portfolio class with empty state."""
        self._state = {
            'returns': {
                'simple_return': None,
                'log_return': None,
                'holding_period_return': None,
                'annualized_return': None,
                'time_weighted_return': None,
                'money_weighted_return': None,
                'arithmetic_return': None,
                'geometric_return': None,
                'total_return_index': None,
                'dollar_weighted_return': None,
                'modified_dietz_return': None,
                'linked_modified_dietz_return': None,
                'leveraged_return': None,
                'market_neutral_return': None,
                'beta_adjusted_return': None,
                'long_short_equity_return': None
            },
            'risk_adjusted': {
                'sharpe_ratio': None,
                'information_ratio': None,
                'capm_alpha': None,
                'benchmark_alpha': None,
                'multifactor_alpha': None,
                'treynor_ratio': None,
                'sortino_ratio': None,
                'calmar_ratio': None,
                'omega_ratio': None,
                'kelly_criterion': None
            },
            'risk': {
                'standard_deviation': None,
                'semi_standard_deviation': None,
                'tracking_error': None,
                'capm_beta': None,
                'value_at_risk': None,
                'conditional_value_at_risk': None,
                'max_drawdown': None,
                'drawdown_duration': None
            }
        }
    
    @property
    def health(self) -> Dict[str, Union[float, str, Dict[str, Dict[str, Union[Dict[str, Optional[float]], float, str]]]]]:
        """
        Calculate and return the overall health of the portfolio based on stored metrics.
        
        Returns
        -------
        dict
            Dictionary containing:
            - overall_score: Float between 0 and 100
            - status: String indicating health status
            - components: Dictionary of component scores and metrics
                - returns: Return metrics and score
                - risk_adjusted: Risk-adjusted performance metrics and score
                - risk: Risk metrics and score
        """
        health_score = 0.0
        components = {}
        
        # Returns Component (25% weight)
        returns_score = 0.0
        returns_count = 0
        if self._state['returns']['simple_return'] is not None:
            return_score = min(100, max(0, (self._state['returns']['simple_return'] * 100) + 50))
            returns_score += return_score
            returns_count += 1
        
        if self._state['returns']['geometric_return'] is not None:
            geo_score = min(100, max(0, (self._state['returns']['geometric_return'] * 100) + 50))
            returns_score += geo_score
            returns_count += 1
            
        if self._state['returns']['time_weighted_return'] is not None:
            twrr_score = min(100, max(0, (self._state['returns']['time_weighted_return'] * 100) + 50))
            returns_score += twrr_score
            returns_count += 1
            
        if returns_count > 0:
            returns_score = returns_score / returns_count
            health_score += returns_score * 0.25
            components['returns'] = {
                'score': returns_score,
                'status': 'Excellent' if returns_score >= 90 else 'Good' if returns_score >= 75 else 'Fair' if returns_score >= 60 else 'Poor' if returns_score >= 45 else 'Critical',
                'metrics': {
                    'simple_return': self._state['returns']['simple_return'],
                    'geometric_return': self._state['returns']['geometric_return'],
                    'time_weighted_return': self._state['returns']['time_weighted_return'],
                    'arithmetic_return': self._state['returns']['arithmetic_return'],
                    'total_return_index': self._state['returns']['total_return_index']
                }
            }
        
        # Risk-Adjusted Performance Component (35% weight)
        risk_adj_score = 0.0
        risk_adj_count = 0
        
        if self._state['risk_adjusted']['sharpe_ratio'] is not None:
            sharpe_score = min(100, max(0, (self._state['risk_adjusted']['sharpe_ratio'] * 20) + 50))
            risk_adj_score += sharpe_score
            risk_adj_count += 1
            
        if self._state['risk_adjusted']['sortino_ratio'] is not None:
            sortino_score = min(100, max(0, (self._state['risk_adjusted']['sortino_ratio'] * 20) + 50))
            risk_adj_score += sortino_score
            risk_adj_count += 1
            
        if self._state['risk_adjusted']['information_ratio'] is not None:
            info_score = min(100, max(0, (self._state['risk_adjusted']['information_ratio'] * 20) + 50))
            risk_adj_score += info_score
            risk_adj_count += 1
            
        if self._state['risk_adjusted']['calmar_ratio'] is not None:
            calmar_score = min(100, max(0, (self._state['risk_adjusted']['calmar_ratio'] * 20) + 50))
            risk_adj_score += calmar_score
            risk_adj_count += 1
            
        if self._state['risk_adjusted']['treynor_ratio'] is not None:
            treynor_score = min(100, max(0, (self._state['risk_adjusted']['treynor_ratio'] * 20) + 50))
            risk_adj_score += treynor_score
            risk_adj_count += 1
            
        if risk_adj_count > 0:
            risk_adj_score = risk_adj_score / risk_adj_count
            health_score += risk_adj_score * 0.35
            components['risk_adjusted'] = {
                'score': risk_adj_score,
                'status': 'Excellent' if risk_adj_score >= 80 else 'Good' if risk_adj_score >= 60 else 'Fair' if risk_adj_score >= 40 else 'Poor' if risk_adj_score >= 20 else 'Critical',
                'metrics': {
                    'sharpe_ratio': self._state['risk_adjusted']['sharpe_ratio'],
                    'sortino_ratio': self._state['risk_adjusted']['sortino_ratio'],
                    'information_ratio': self._state['risk_adjusted']['information_ratio'],
                    'calmar_ratio': self._state['risk_adjusted']['calmar_ratio'],
                    'treynor_ratio': self._state['risk_adjusted']['treynor_ratio'],
                    'omega_ratio': self._state['risk_adjusted']['omega_ratio'],
                    'kelly_criterion': self._state['risk_adjusted']['kelly_criterion']
                }
            }
        
        # Risk Component (40% weight)
        risk_score = 0.0
        risk_components = 0
        
        if self._state['risk']['standard_deviation'] is not None:
            vol_score = max(0, min(100, 100 - (self._state['risk']['standard_deviation'] * 100)))
            risk_score += vol_score
            risk_components += 1
        
        if self._state['risk']['value_at_risk'] is not None:
            var_score = max(0, min(100, 100 - (self._state['risk']['value_at_risk'] * 100)))
            risk_score += var_score
            risk_components += 1
        
        if self._state['risk']['conditional_value_at_risk'] is not None:
            cvar_score = max(0, min(100, 100 - (self._state['risk']['conditional_value_at_risk'] * 100)))
            risk_score += cvar_score
            risk_components += 1
            
        if self._state['risk']['max_drawdown'] is not None:
            dd_score = max(0, min(100, 100 - (self._state['risk']['max_drawdown'] * 100)))
            risk_score += dd_score
            risk_components += 1
            
        if self._state['risk']['semi_standard_deviation'] is not None:
            semi_std_score = max(0, min(100, 100 - (self._state['risk']['semi_standard_deviation'] * 100)))
            risk_score += semi_std_score
            risk_components += 1
        
        if risk_components > 0:
            risk_score = risk_score / risk_components
            health_score += risk_score * 0.40
            components['risk'] = {
                'score': risk_score,
                'status': 'Excellent' if risk_score >= 85 else 'Good' if risk_score >= 70 else 'Fair' if risk_score >= 55 else 'Poor' if risk_score >= 40 else 'Critical',
                'metrics': {
                    'standard_deviation': self._state['risk']['standard_deviation'],
                    'semi_standard_deviation': self._state['risk']['semi_standard_deviation'],
                    'value_at_risk': self._state['risk']['value_at_risk'],
                    'conditional_value_at_risk': self._state['risk']['conditional_value_at_risk'],
                    'max_drawdown': self._state['risk']['max_drawdown'],
                    'drawdown_duration': self._state['risk']['drawdown_duration'],
                    'tracking_error': self._state['risk']['tracking_error'],
                    'capm_beta': self._state['risk']['capm_beta']
                }
            }
        
        return {
            'overall_score': health_score,
            'status': 'Excellent' if health_score >= 90 else 'Good' if health_score >= 75 else 'Fair' if health_score >= 60 else 'Poor' if health_score >= 45 else 'Critical',
            'components': components
        }
    
    def simple_return(
        self,
        end_value: Union[float, List[float], np.ndarray],
        start_value: Union[float, List[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        result = simple_return(end_value, start_value)
        if isinstance(result, (int, float)):
            self._state['returns']['simple_return'] = result
        return result
    
    def log_return(
        self,
        end_value: Union[float, List[float], np.ndarray],
        start_value: Union[float, List[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        result = log_return(end_value, start_value)
        if isinstance(result, (int, float)):
            self._state['returns']['log_return'] = result
        return result
    
    def holding_period_return(
        self,
        prices: Union[List[float], np.ndarray],
        dividends: Optional[Union[List[float], np.ndarray]] = None
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
        result = holding_period_return(prices, dividends)
        self._state['returns']['holding_period_return'] = result
        return result
    
    def annualized_return(
        self,
        total_return: Union[float, List[float], np.ndarray],
        years: Union[float, List[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        result = annualized_return(total_return, years)
        if isinstance(result, (int, float)):
            self._state['returns']['annualized_return'] = result
        return result
    
    def time_weighted_return(
        self,
        period_returns: Union[List[float], np.ndarray]
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
        result = time_weighted_return(period_returns)
        self._state['returns']['time_weighted_return'] = result
        return result
    
    def money_weighted_return(
        self,
        cash_flows: Union[List[float], np.ndarray],
        cash_flow_times: Union[List[float], np.ndarray],
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
        result = money_weighted_return(cash_flows, cash_flow_times, final_value, initial_value, max_iterations, tolerance)
        self._state['returns']['money_weighted_return'] = result
        return result
    
    def arithmetic_return(self, prices: Union[List[float], np.ndarray]) -> float:
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
        result = arithmetic_return(prices)
        self._state['returns']['arithmetic_return'] = result
        return result
    
    def geometric_return(self, prices: Union[List[float], np.ndarray]) -> float:
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
        result = geometric_return(prices)
        self._state['returns']['geometric_return'] = result
        return result
    
    def total_return_index(
        self,
        prices: Union[List[float], np.ndarray],
        dividends: Optional[Union[List[float], np.ndarray]] = None
    ) -> np.ndarray:
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
        result = total_return_index(prices, dividends)
        self._state['returns']['total_return_index'] = result[-1] if len(result) > 0 else None
        return result
    
    def dollar_weighted_return(
        self,
        cash_flows: Union[List[float], np.ndarray],
        cash_flow_dates: Union[List[float], np.ndarray],
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
        result = dollar_weighted_return(cash_flows, cash_flow_dates, end_value)
        self._state['returns']['dollar_weighted_return'] = result
        return result
    
    def modified_dietz_return(
        self,
        start_value: float,
        end_value: float,
        cash_flows: Union[List[float], np.ndarray],
        cash_flow_days: Union[List[float], np.ndarray],
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
        result = modified_dietz_return(start_value, end_value, cash_flows, cash_flow_days, total_days)
        self._state['returns']['modified_dietz_return'] = result
        return result
    
    def linked_modified_dietz_return(
        self,
        period_returns: Union[List[float], np.ndarray]
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
        result = linked_modified_dietz_return(period_returns)
        self._state['returns']['linked_modified_dietz_return'] = result
        return result
    
    def leveraged_return(
        self,
        unleveraged_return: Union[float, List[float], np.ndarray],
        leverage_ratio: Union[float, List[float], np.ndarray],
        borrowing_rate: Union[float, List[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        result = leveraged_return(unleveraged_return, leverage_ratio, borrowing_rate)
        if isinstance(result, (int, float)):
            self._state['returns']['leveraged_return'] = result
        return result
    
    def market_neutral_return(
        self,
        long_return: Union[float, List[float], np.ndarray],
        short_return: Union[float, List[float], np.ndarray],
        long_weight: Union[float, List[float], np.ndarray] = 0.5,
        short_weight: Union[float, List[float], np.ndarray] = 0.5,
        short_borrowing_cost: Union[float, List[float], np.ndarray] = 0.0
    ) -> Union[float, np.ndarray]:
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
        result = market_neutral_return(long_return, short_return, long_weight, short_weight, short_borrowing_cost)
        if isinstance(result, (int, float)):
            self._state['returns']['market_neutral_return'] = result
        return result
    
    def beta_adjusted_return(
            self, 
            portfolio_return: Union[float, List[float], np.ndarray],
            benchmark_return: Union[float, List[float], np.ndarray],
            portfolio_beta: Union[float, List[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        result = beta_adjusted_return(portfolio_return, benchmark_return, portfolio_beta)
        if isinstance(result, (int, float)):
            self._state['returns']['beta_adjusted_return'] = result
        return result
    
    def long_short_equity_return(
            self, 
            long_portfolio_return: Union[float, List[float], np.ndarray],
            short_portfolio_return: Union[float, List[float], np.ndarray],
            long_exposure: Union[float, List[float], np.ndarray],
            short_exposure: Union[float, List[float], np.ndarray],
            risk_free_rate: Union[float, List[float], np.ndarray] = 0.0,
            short_rebate: Union[float, List[float], np.ndarray] = 0.0
            ) -> Union[float, np.ndarray]:
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
        [0.14, 0.15]
        >>> long_short_equity_return(0.10, -0.05, [1.0, 0.8], [0.5, 0.4], [0.02, 0.03], 0.01)
        [0.14 , 0.122]
        """
        result = long_short_equity_return(long_portfolio_return, short_portfolio_return, 
                                          long_exposure, short_exposure, risk_free_rate, short_rebate)
        if isinstance(result, (int, float)):
            self._state['returns']['long_short_equity_return'] = result
        return result
    
    def benchmark_alpha(
        self,
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
        result = benchmark_alpha(returns, benchmark_returns)
        self._state['risk_adjusted']['benchmark_alpha'] = result
        return result
    
    def multifactor_alpha(
        self,
        returns: Union[List[float], np.ndarray],
        factor_returns: Union[List[List[float]], np.ndarray],
        risk_free_rate: float = 0.0
    ) -> float:
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
        alpha, _, _, _, _ = multifactor_alpha(returns, factor_returns, risk_free_rate)
        self._state['risk_adjusted']['multifactor_alpha'] = alpha
        return alpha
    
    def treynor_ratio(self,
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
        result = treynor_ratio(returns, benchmark_returns, risk_free_rate, annualization_factor)
        self._state['risk_adjusted']['treynor_ratio'] = result
        return result
    
    def sortino_ratio(self,
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
        result = sortino_ratio(returns, risk_free_rate, target_return, annualization_factor)
        self._state['risk_adjusted']['sortino_ratio'] = result
        return result
    
    def calmar_ratio(self, returns: Union[List[float], np.ndarray],
            max_drawdown: Optional[float] = None,
            annualization_factor: float = 1.0) -> float:
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
        result = calmar_ratio(returns, max_drawdown, annualization_factor)
        self._state['risk_adjusted']['calmar_ratio'] = result
        return result
    
    def omega_ratio(self,
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
        result = omega_ratio(returns, threshold, annualization_factor)
        self._state['risk_adjusted']['omega_ratio'] = result
        return result
    
    def kelly_criterion(self,
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
        result = kelly_criterion(returns, risk_free_rate)
        self._state['risk_adjusted']['kelly_criterion'] = result
        return result
    
    def semi_standard_deviation(self,
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
        result = semi_standard_deviation(returns, threshold, annualize, periods_per_year)
        self._state['risk']['semi_standard_deviation'] = result
        return result
    
    def tracking_error(self, 
                  portfolio_returns: Union[List[float], np.ndarray], 
                  benchmark_returns: Union[List[float], np.ndarray], 
                  annualize: bool = False, 
                  periods_per_year: int = 252) -> float:
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
        result = tracking_error(portfolio_returns, benchmark_returns, annualize, periods_per_year)
        self._state['risk']['tracking_error'] = result
        return result
    
    def capm_beta(self, portfolio_returns: Union[List[float], np.ndarray], 
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
        result = capm_beta(portfolio_returns, market_returns)
        self._state['risk']['capm_beta'] = result
        return result
    
    def covariance_matrix(
        self,
        returns_matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
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
        return np.array(covariance_matrix(returns_matrix))
    
    def correlation_matrix(self, 
                           returns_matrix: Union[List[List[float]], np.ndarray], 
                           ) -> np.ndarray:
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
        return correlation_matrix(returns_matrix)
    
    def conditional_value_at_risk(self, returns: Union[List[float], np.ndarray], confidence_level: float = 0.95,
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
        result = conditional_value_at_risk(returns, confidence_level, method, current_value)
        self._state['risk']['conditional_value_at_risk'] = result
        return result
    
    def value_at_risk(
        self,
        returns: Union[List[float], np.ndarray],
        confidence_level: float = 0.95,
        method: str = 'historical',
        parametric_mean: Optional[float] = None,
        parametric_std: Optional[float] = None,
        current_value: float = 1.0
    ) -> float:
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
        result = value_at_risk(returns, confidence_level, method, parametric_mean, parametric_std, current_value)
        self._state['risk']['value_at_risk'] = result
        return result
    
    def drawdown(
        self,
        returns: Union[List[float], np.ndarray],
        as_list: bool = False
    ) -> Union[Tuple[np.ndarray, float, int, int], Tuple[List[float], float, int, int]]:
        """Calculate drawdown metrics."""
        result = drawdown(returns, as_list)
        self._state['risk']['max_drawdown'] = result[1]
        self._state['risk']['drawdown_duration'] = result[3] - result[2]
        return result 