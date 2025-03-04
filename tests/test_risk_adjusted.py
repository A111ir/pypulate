"""Tests for risk-adjusted performance functions."""

import pytest
import numpy as np
from pypulate.portfolio.risk_adjusted import (
    sharpe_ratio,
    information_ratio,
    capm_alpha,
    benchmark_alpha,
    multifactor_alpha,
    treynor_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    kelly_criterion
)


def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Test with single risk-free rate
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    risk_free = 0.001
    
    sharpe = sharpe_ratio(returns, risk_free, annualization_factor=252)
    assert sharpe == pytest.approx(13.162446581088183, abs=1e-4)
    
    # Test with array of risk-free rates
    risk_free_array = [0.001, 0.002]
    sharpe_array = sharpe_ratio(returns, risk_free_array, annualization_factor=252)
    expected = [13.16244658, 11.96586053]
    np.testing.assert_array_almost_equal(sharpe_array, expected, decimal=4)


def test_information_ratio():
    """Test Information ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    benchmark = [0.005, 0.01, -0.005, 0.02, 0.005]
    
    ir = information_ratio(returns, benchmark, annualization_factor=252)
    assert ir == pytest.approx(14.4914, abs=1e-4)


def test_capm_alpha():
    """Test CAPM alpha calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    benchmark = [0.005, 0.01, -0.005, 0.02, 0.005]
    risk_free = 0.001
    
    alpha, beta, r_squared, p_value, std_err = capm_alpha(
        returns, benchmark, risk_free
    )
    
    # Test alpha value
    assert alpha == pytest.approx(0.00136363, abs=1e-4)
    # Test beta value
    assert beta == pytest.approx(1.60606, abs=1e-1)
    # Test R-squared
    assert r_squared == pytest.approx(0.967286, abs=1e-1)
    # Test p-value and standard error are reasonable
    assert 0 <= p_value <= 1
    assert std_err > 0


def test_benchmark_alpha():
    """Test benchmark alpha calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    benchmark = [0.005, 0.01, -0.005, 0.02, 0.005]
    
    alpha = benchmark_alpha(returns, benchmark)
    assert alpha == pytest.approx(0.005, abs=1e-4)


def test_multifactor_alpha():
    """Test multifactor alpha calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    factors = [
        [0.005, 0.01, -0.005, 0.02, 0.005],  # Market
        [0.002, 0.003, -0.001, 0.004, 0.001],  # Size
        [0.001, 0.002, -0.002, 0.003, 0.002]   # Value
    ]
    risk_free = 0.001
    
    alpha, betas, r_squared, p_value, std_err = multifactor_alpha(
        returns, factors, risk_free
    )
    
    # Test alpha value
    assert alpha == pytest.approx(-0.00109, abs=1e-4)
    # Test betas array
    assert betas == pytest.approx([0.61538462, 2.98076923, 2.01923077], abs=1e-4)
    # Test R-squared
    assert r_squared == pytest.approx(0.99781, abs=1e-1)
    # Test p-value and standard error are reasonable
    assert 0 <= p_value <= 1
    assert std_err > 0


def test_treynor_ratio():
    """Test Treynor ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    benchmark = [0.005, 0.01, -0.005, 0.02, 0.005]
    risk_free = 0.001
    
    treynor = treynor_ratio(returns, benchmark, risk_free, annualization_factor=252)
    assert treynor == pytest.approx(1.3808, abs=1e-4)


def test_sortino_ratio():
    """Test Sortino ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    risk_free = 0.001
    target = 0.0
    
    sortino = sortino_ratio(returns, risk_free, target, annualization_factor=252)
    assert sortino == pytest.approx(17.4620, abs=1e-4)
    
    # Test case with no returns below target
    returns_no_downside = [0.01, 0.02, 0.01, 0.03, 0.01]
    assert sortino_ratio(returns_no_downside, risk_free) == float('inf')


def test_calmar_ratio():
    """Test Calmar ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    max_dd = 0.15
    
    # Test with provided max drawdown
    calmar = calmar_ratio(returns, max_dd, annualization_factor=252)
    assert calmar == pytest.approx(20.16, abs=1e-2)
    
    # Test with calculated max drawdown
    calmar = calmar_ratio(returns, annualization_factor=252)
    assert calmar > 0
    
    # Test case with no drawdown
    returns_no_dd = [0.01, 0.02, 0.03, 0.04, 0.05]
    assert calmar_ratio(returns_no_dd) == float('inf')


def test_omega_ratio():
    """Test Omega ratio calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    threshold = 0.005
    
    omega = omega_ratio(returns, threshold)
    assert omega == pytest.approx(3.3333, abs=1e-4)
    
    # Test case with no returns below threshold
    returns_above = [0.01, 0.02, 0.01, 0.03, 0.01]
    assert omega_ratio(returns_above, 0.0) == float('inf')


def test_kelly_criterion():
    """Test Kelly criterion calculation."""
    returns = [0.01, 0.02, -0.01, 0.03, 0.01]
    risk_free = 0.001
    
    kelly = kelly_criterion(returns, risk_free)
    assert kelly == pytest.approx(0.8292, abs=1e-4)

