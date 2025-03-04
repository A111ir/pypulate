"""
Portfolio Allocation Module

This module provides various portfolio optimization methods including
Mean-Variance Optimization, Minimum Variance Portfolio, Maximum Sharpe Ratio,
Hierarchical Risk Parity, Black-Litterman, Kelly Criterion, and other common
portfolio optimization techniques.
"""

from .optimization import (
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

__all__ = [
    'mean_variance_optimization',
    'minimum_variance_portfolio',
    'maximum_sharpe_ratio',
    'risk_parity_portfolio',
    'maximum_diversification_portfolio',
    'equal_weight_portfolio',
    'market_cap_weight_portfolio',
    'hierarchical_risk_parity',
    'black_litterman',
    'kelly_criterion_optimization'
]
