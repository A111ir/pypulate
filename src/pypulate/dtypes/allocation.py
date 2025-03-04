"""
Portfolio Allocation Module

This module provides a class for portfolio allocation optimization,
including various methods like Mean-Variance Optimization, Minimum Variance,
Maximum Sharpe Ratio, Hierarchical Risk Parity, Black-Litterman, Kelly Criterion,
and other common portfolio optimization techniques.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from ..allocation.optimization import (
    mean_variance_optimization,
    minimum_variance_portfolio,
    maximum_sharpe_ratio,
    risk_parity_portfolio,
    maximum_diversification_portfolio,
    equal_weight_portfolio,
    market_cap_weight_portfolio,
    hierarchical_risk_parity,
    black_litterman,
    kelly_criterion_optimization
)


class Allocation:
    """
    A class for portfolio allocation optimization.
    
    This class provides methods for various portfolio optimization techniques
    including Mean-Variance Optimization, Minimum Variance Portfolio,
    Maximum Sharpe Ratio, and other common portfolio optimization methods.
    
    Examples
    --------
    >>> from pypulate.dtypes import Allocation
    >>> allocation = Allocation()
    >>> weights, ret, risk = allocation.mean_variance(returns_matrix)
    >>> weights = allocation.equal_weight(returns_matrix)
    """
    
    def __init__(self):
        """Initialize the Allocation class with empty state."""
        self._state = {
            'weights': None,
            'portfolio_return': None,
            'portfolio_risk': None,
            'method': None,
            'constraints': None
        }
    
    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get the current portfolio weights."""
        return self._state['weights']
    
    @property
    def portfolio_return(self) -> Optional[float]:
        """Get the current portfolio return."""
        return self._state['portfolio_return']
    
    @property
    def portfolio_risk(self) -> Optional[float]:
        """Get the current portfolio risk."""
        return self._state['portfolio_risk']
    
    @property
    def method(self) -> Optional[str]:
        """Get the current optimization method used."""
        return self._state['method']
    
    def mean_variance(
        self,
        returns: np.ndarray,
        target_return: Optional[float] = None,
        risk_free_rate: float = 0.0,
        constraints: Optional[List[dict]] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Perform Mean-Variance Optimization to find optimal portfolio weights.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        target_return : float, optional
            Target portfolio return. If None, maximizes Sharpe ratio
        risk_free_rate : float, default 0.0
            Risk-free rate for Sharpe ratio calculation
        constraints : list of dict, optional
            List of constraints for the optimization problem
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = mean_variance_optimization(
            returns, target_return, risk_free_rate, constraints
        )
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'mean_variance',
            'constraints': constraints
        })
        return weights, ret, risk
    
    def minimum_variance(
        self,
        returns: np.ndarray,
        constraints: Optional[List[dict]] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Find the portfolio with minimum variance.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        constraints : list of dict, optional
            List of constraints for the optimization problem
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = minimum_variance_portfolio(returns, constraints)
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'minimum_variance',
            'constraints': constraints
        })
        return weights, ret, risk
    
    def maximum_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        constraints: Optional[List[dict]] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Find the portfolio with maximum Sharpe ratio.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        risk_free_rate : float, default 0.0
            Risk-free rate for Sharpe ratio calculation
        constraints : list of dict, optional
            List of constraints for the optimization problem
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = maximum_sharpe_ratio(returns, risk_free_rate, constraints)
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'maximum_sharpe',
            'constraints': constraints
        })
        return weights, ret, risk
    
    def risk_parity(
        self,
        returns: np.ndarray,
        constraints: Optional[List[dict]] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Find the portfolio where risk is equally distributed across assets.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        constraints : list of dict, optional
            List of constraints for the optimization problem
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = risk_parity_portfolio(returns, constraints)
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'risk_parity',
            'constraints': constraints
        })
        return weights, ret, risk
    
    def maximum_diversification(
        self,
        returns: np.ndarray,
        constraints: Optional[List[dict]] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Find the portfolio that maximizes diversification ratio.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        constraints : list of dict, optional
            List of constraints for the optimization problem
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = maximum_diversification_portfolio(returns, constraints)
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'maximum_diversification',
            'constraints': constraints
        })
        return weights, ret, risk
    
    def equal_weight(
        self,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Create an equal-weighted portfolio.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
            
        Returns
        -------
        tuple
            (weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = equal_weight_portfolio(returns)
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'equal_weight',
            'constraints': None
        })
        return weights, ret, risk
    
    def market_cap_weight(
        self,
        market_caps: np.ndarray
    ) -> np.ndarray:
        """
        Create a market-cap weighted portfolio.
        
        Parameters
        ----------
        market_caps : np.ndarray
            Array of market capitalizations for each asset
            
        Returns
        -------
        np.ndarray
            Market-cap weighted portfolio weights
        """
        weights = market_cap_weight_portfolio(market_caps)
        self._state.update({
            'weights': weights,
            'portfolio_return': None,
            'portfolio_risk': None,
            'method': 'market_cap_weight',
            'constraints': None
        })
        return weights
    
    def get_portfolio_metrics(self) -> Dict[str, Optional[Union[np.ndarray, float, str]]]:
        """
        Get the current portfolio metrics.
        
        Returns
        -------
        dict
            Dictionary containing:
            - weights: Current portfolio weights
            - portfolio_return: Current portfolio return
            - portfolio_risk: Current portfolio risk
            - method: Current optimization method
            - constraints: Current optimization constraints
        """
        return self._state.copy()
    
    def hierarchical_risk_parity(
        self,
        returns: np.ndarray,
        linkage_method: str = 'single',
        distance_metric: str = 'euclidean'
    ) -> Tuple[np.ndarray, float, float]:
        """
        Implement Hierarchical Risk Parity (HRP) portfolio optimization.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        linkage_method : str, default 'single'
            Linkage method for hierarchical clustering ('single', 'complete', 'average', 'ward')
        distance_metric : str, default 'euclidean'
            Distance metric for clustering ('euclidean', 'correlation')
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = hierarchical_risk_parity(
            returns, linkage_method, distance_metric
        )
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'hierarchical_risk_parity',
            'constraints': None
        })
        return weights, ret, risk
    
    def black_litterman(
        self,
        returns: np.ndarray,
        market_caps: np.ndarray,
        views: Dict[int, float],
        view_confidences: Dict[int, float],
        tau: float = 0.05,
        risk_free_rate: float = 0.0
    ) -> Tuple[np.ndarray, float, float]:
        """
        Implement Black-Litterman portfolio optimization.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        market_caps : np.ndarray
            Array of market capitalizations for each asset
        views : dict
            Dictionary mapping asset indices to expected returns
        view_confidences : dict
            Dictionary mapping asset indices to confidence levels (0-1)
        tau : float, default 0.05
            Uncertainty in the prior distribution
        risk_free_rate : float, default 0.0
            Risk-free rate
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
        """
        weights, ret, risk = black_litterman(
            returns, market_caps, views, view_confidences, tau, risk_free_rate
        )
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'black_litterman',
            'constraints': None
        })
        return weights, ret, risk
    
    def kelly_criterion(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        constraints: Optional[List[dict]] = None,
        kelly_fraction: float = 1.0
    ) -> Tuple[np.ndarray, float, float]:
        """
        Implement Kelly Criterion portfolio optimization.
        
        The Kelly Criterion determines the optimal fraction of capital to allocate
        to each investment to maximize long-term growth rate.
        
        Parameters
        ----------
        returns : np.ndarray
            Matrix of asset returns where each column represents an asset
        risk_free_rate : float, default 0.0
            Risk-free rate
        constraints : list of dict, optional
            List of constraints for the optimization problem
            
        Returns
        -------
        tuple
            (optimal_weights, portfolio_return, portfolio_risk)
            
        Notes
        -----
        The Kelly Criterion maximizes the expected logarithmic growth rate of wealth.
        The full Kelly Criterion can be aggressive, so practitioners often use a
        fraction of the Kelly Criterion (e.g., half-Kelly) for more conservative
        position sizing.
        """
        weights, ret, risk = kelly_criterion_optimization(
            returns, risk_free_rate, constraints, kelly_fraction
        )
        self._state.update({
            'weights': weights,
            'portfolio_return': ret,
            'portfolio_risk': risk,
            'method': 'kelly_criterion',
            'constraints': constraints
        })
        return weights, ret, risk 