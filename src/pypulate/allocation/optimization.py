"""
Portfolio Optimization Module

This module provides various portfolio optimization methods including
Mean-Variance Optimization, Minimum Variance Portfolio, Maximum Sharpe Ratio,
Hierarchical Risk Parity, Black-Litterman, Kelly Criterion, and other common
portfolio optimization techniques.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from ..portfolio.risk_measurement import covariance_matrix, standard_deviation

def mean_variance_optimization(
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
    n_assets = returns.shape[1]
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = covariance_matrix(returns)
    
    def objective(weights: np.ndarray) -> float:
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if target_return is None:
            return -((portfolio_return - risk_free_rate) / portfolio_risk)
        else:
            return portfolio_risk
    
    if constraints is None:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
            {'type': 'ineq', 'fun': lambda x: x} 
        ]
    
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(mean_returns * x) - target_return
        })
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    portfolio_return = np.sum(mean_returns * optimal_weights)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    
    return optimal_weights, portfolio_return, portfolio_risk

def minimum_variance_portfolio(
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
    n_assets = returns.shape[1]
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = covariance_matrix(returns)
    
    def objective(weights: np.ndarray) -> float:
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    if constraints is None:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    portfolio_return = np.sum(mean_returns * optimal_weights)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    
    return optimal_weights, portfolio_return, portfolio_risk

def maximum_sharpe_ratio(
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
    return mean_variance_optimization(returns, None, risk_free_rate, constraints)

def risk_parity_portfolio(
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
    n_assets = returns.shape[1]
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = covariance_matrix(returns)
    
    def objective(weights: np.ndarray) -> float:
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        risk_contributions = weights * (np.dot(cov_matrix, weights) / portfolio_risk)
        return np.sum((risk_contributions - portfolio_risk/n_assets) ** 2)
    
    if constraints is None:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
            {'type': 'ineq', 'fun': lambda x: x} 
        ]
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    portfolio_return = np.sum(mean_returns * optimal_weights)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    
    return optimal_weights, portfolio_return, portfolio_risk

def maximum_diversification_portfolio(
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
    n_assets = returns.shape[1]
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = covariance_matrix(returns)
    vols = np.sqrt(np.diag(cov_matrix))
    
    def objective(weights: np.ndarray) -> float:
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        weighted_vol = np.sum(weights * vols)
        return -weighted_vol / portfolio_risk  
    
    if constraints is None:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
            {'type': 'ineq', 'fun': lambda x: x} 
        ]
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    portfolio_return = np.sum(mean_returns * optimal_weights)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    
    return optimal_weights, portfolio_return, portfolio_risk

def equal_weight_portfolio(
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
    n_assets = returns.shape[1]
    weights = np.array([1/n_assets] * n_assets)
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = covariance_matrix(returns)
    
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return weights, portfolio_return, portfolio_risk

def market_cap_weight_portfolio(
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
    total_market_cap = np.sum(market_caps)
    return market_caps / total_market_cap

def hierarchical_risk_parity(
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
    cov_matrix = covariance_matrix(returns)
    mean_returns = np.mean(returns, axis=0)
    
    vols = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(vols, vols)
    
    corr_matrix = np.clip(corr_matrix, -1, 1)
    
    if distance_metric == 'correlation':
        dist_matrix = np.sqrt(np.clip(0.5 * (1 - corr_matrix), 0, 1))
    else: 
        dist_matrix = np.zeros((corr_matrix.shape[0], corr_matrix.shape[0]))
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[0]):
                dist_matrix[i, j] = np.sqrt(np.sum((corr_matrix[i] - corr_matrix[j])**2))
    
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)
    
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method=linkage_method)
    
    n_clusters = 1
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    weights = np.zeros(len(returns[0]))
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_cov = cov_matrix[cluster_indices][:, cluster_indices]
        
        epsilon = 1e-6
        cluster_cov = cluster_cov + np.eye(cluster_cov.shape[0]) * epsilon
        
        try:
            cluster_weights = np.linalg.inv(cluster_cov).sum(axis=1)
            cluster_weights = cluster_weights / cluster_weights.sum()
            weights[cluster_indices] = cluster_weights
        except np.linalg.LinAlgError:
            weights[cluster_indices] = 1.0 / len(cluster_indices)
    
    weights = weights / weights.sum()
    
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return weights, portfolio_return, portfolio_risk

def black_litterman(
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
    cov_matrix = covariance_matrix(returns)
    market_weights = market_caps / np.sum(market_caps)
    risk_aversion = (np.mean(returns) - risk_free_rate) / np.var(returns)
    equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_weights)
    
    n_assets = len(returns[0])
    P = np.zeros((len(views), n_assets))
    Q = np.zeros(len(views))
    Omega = np.zeros((len(views), len(views)))
    
    for i, (asset_idx, view_return) in enumerate(views.items()):
        P[i, asset_idx] = 1
        Q[i] = view_return
        Omega[i, i] = 1 / view_confidences[asset_idx]
    
    tau_cov = tau * cov_matrix
    BL_returns = equilibrium_returns + np.dot(
        np.dot(tau_cov, P.T),
        np.linalg.inv(np.dot(np.dot(P, tau_cov), P.T) + Omega)
    ).dot(Q - np.dot(P, equilibrium_returns))
    
    BL_cov = cov_matrix + tau_cov - np.dot(
        np.dot(tau_cov, P.T),
        np.linalg.inv(np.dot(np.dot(P, tau_cov), P.T) + Omega)
    ).dot(np.dot(P, tau_cov))
    
    def objective(weights: np.ndarray) -> float:
        portfolio_return = np.sum(BL_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(BL_cov, weights)))
        return -((portfolio_return - risk_free_rate) / portfolio_risk)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
        {'type': 'ineq', 'fun': lambda x: x} 
    ]
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    portfolio_return = np.sum(BL_returns * optimal_weights)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(BL_cov, optimal_weights)))
    
    return optimal_weights, portfolio_return, portfolio_risk

def kelly_criterion_optimization(
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
    kelly_fraction : float, default 1.0
        Fraction of Kelly Criterion to use (e.g., 0.5 for half-Kelly)
        
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
    n_assets = returns.shape[1]
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = covariance_matrix(returns)
    
    epsilon = 1e-8
    cov_matrix += np.eye(n_assets) * epsilon
    
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        excess_returns = mean_returns - risk_free_rate
        optimal_weights = np.dot(inv_cov_matrix, excess_returns)
        
        optimal_weights *= kelly_fraction
        
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        if constraints is not None:
            def objective(weights: np.ndarray) -> float:
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return - 0.5 * portfolio_risk**2)
            
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = optimal_weights
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            optimal_weights = result.x
        
        portfolio_return = np.sum(mean_returns * optimal_weights)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        
        return optimal_weights, portfolio_return, portfolio_risk
        
    except np.linalg.LinAlgError:
        def objective(weights: np.ndarray) -> float:
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - 0.5 * portfolio_risk**2)
        
        if constraints is None:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, 
                {'type': 'ineq', 'fun': lambda x: x}  
            ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        portfolio_return = np.sum(mean_returns * optimal_weights)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        
        return optimal_weights, portfolio_return, portfolio_risk 