"""
Tests for the portfolio risk-adjusted performance measurement functions.
"""

import numpy as np
import pytest
from pypulate.portfolio.risk_adjusted import (
    sharpe_ratio, information_ratio, capm_alpha, benchmark_alpha,
    multifactor_alpha, treynor_ratio, sortino_ratio, calmar_ratio,
    omega_ratio
)


class TestRiskAdjustedMeasures:
    """Test class for portfolio risk-adjusted performance measures."""
    
    def test_sharpe_ratio_scalar(self):
        """Test sharpe_ratio with scalar risk-free rate."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        
        # Basic test
        assert sharpe_ratio(returns, 0.001) == pytest.approx(0.8291562, abs=1e-6)
        
        # Annualized
        assert sharpe_ratio(returns, 0.001, 252) == pytest.approx(13.162447, abs=1e-6)
        
        # Zero risk-free rate
        assert sharpe_ratio(returns, 0.0) == pytest.approx(0.9045340, abs=1e-6)
        
        # Risk-free rate equals mean return
        mean_return = np.mean(returns)
        assert sharpe_ratio(returns, mean_return) == pytest.approx(0.0, abs=1e-10)
    
    def test_sharpe_ratio_array(self):
        """Test sharpe_ratio with array risk-free rate."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        risk_free_rates = [0.001, 0.002]
        
        result = sharpe_ratio(returns, risk_free_rates, 252)
        expected = np.array([13.162447, 11.965861])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
        # NumPy array input
        result = sharpe_ratio(np.array(returns), np.array(risk_free_rates), 252)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
    
    def test_information_ratio(self):
        """Test information_ratio function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        benchmark_returns = [0.005, 0.01, -0.005, 0.02, 0.005]
        
        # Basic test
        assert information_ratio(returns, benchmark_returns) == pytest.approx(0.9128709, abs=1e-6)
        
        # Annualized
        assert information_ratio(returns, benchmark_returns, 252) == pytest.approx(14.4913767, abs=1e-6)
        
        # Same returns (should now return 0.0 instead of NaN)
        assert information_ratio(returns, returns) == 0.0
        
        # NumPy array input
        assert information_ratio(np.array(returns), np.array(benchmark_returns)) == pytest.approx(0.9128709, abs=1e-6)
    
    def test_capm_alpha(self):
        """Test capm_alpha function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        benchmark_returns = [0.005, 0.01, -0.005, 0.02, 0.005]
        
        # Basic test
        alpha, beta, r_squared, p_value, std_err = capm_alpha(returns, benchmark_returns, 0.001)
        
        assert alpha == pytest.approx(0.0013636, abs=1e-6)
        assert beta == pytest.approx(1.6060606, abs=1e-6)
        assert r_squared == pytest.approx(0.9672865, abs=1e-6)
        
        # Test with NumPy array input
        alpha2, beta2, r_squared2, _, _ = capm_alpha(np.array(returns), np.array(benchmark_returns), 0.001)
        assert alpha2 == pytest.approx(alpha, abs=1e-10)
        assert beta2 == pytest.approx(beta, abs=1e-10)
        assert r_squared2 == pytest.approx(r_squared, abs=1e-10)
    
    def test_benchmark_alpha(self):
        """Test benchmark_alpha function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        benchmark_returns = [0.005, 0.01, -0.005, 0.02, 0.005]
        
        # Basic test
        assert benchmark_alpha(returns, benchmark_returns) == pytest.approx(0.005, abs=1e-10)
        
        # Same returns
        assert benchmark_alpha(returns, returns) == pytest.approx(0.0, abs=1e-10)
        
        # Underperformance
        assert benchmark_alpha([0.01, 0.01], [0.02, 0.02]) == pytest.approx(-0.01, abs=1e-10)
    
    def test_multifactor_alpha(self):
        """Test multifactor_alpha function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        
        # Market, Size, and Value factors
        factor_returns = [
            [0.005, 0.01, -0.005, 0.02, 0.005],  # Market
            [0.002, 0.003, -0.001, 0.004, 0.001],  # Size
            [0.001, 0.002, -0.002, 0.003, 0.002]   # Value
        ]
        
        # Basic test
        alpha, betas, r_squared, p_value, std_err = multifactor_alpha(returns, factor_returns, 0.001)
        
        # Since we don't know the exact values, just check the types and shapes
        assert isinstance(alpha, float)
        assert isinstance(betas, np.ndarray)
        assert len(betas) == 3
        assert isinstance(r_squared, float)
        assert 0 <= r_squared <= 1
        
        # Test with transposed factor returns
        factor_returns_transposed = np.array(factor_returns).T
        alpha2, betas2, r_squared2, _, _ = multifactor_alpha(returns, factor_returns_transposed, 0.001)
        
        # Values should be similar
        np.testing.assert_array_almost_equal(betas, betas2, decimal=6)
        assert r_squared == pytest.approx(r_squared2, abs=1e-6)
    
    def test_treynor_ratio(self):
        """Test treynor_ratio function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        benchmark_returns = [0.005, 0.01, -0.005, 0.02, 0.005]
        
        # Basic test
        assert treynor_ratio(returns, benchmark_returns, 0.001) == pytest.approx(0.0054792, abs=1e-6)
        
        # Annualized
        assert treynor_ratio(returns, benchmark_returns, 0.001, 252) == pytest.approx(1.3807698, abs=1e-6)
        
        # Zero risk-free rate
        assert treynor_ratio(returns, benchmark_returns, 0) == pytest.approx(0.0059774, abs=1e-6)
    
    def test_sortino_ratio(self):
        """Test sortino_ratio function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        
        # Basic test
        assert sortino_ratio(returns, 0.001) == pytest.approx(1.1, abs=1e-6)
        
        # Annualized
        assert sortino_ratio(returns, 0.001, 0.0, 252) == pytest.approx(17.4619587, abs=1e-6)
        
        # Higher target return
        assert sortino_ratio(returns, 0.001, 0.01) == pytest.approx(0.55, abs=1e-6)
        
        # No negative returns relative to target
        assert sortino_ratio([0.02, 0.03, 0.04], 0.01, 0.01) == float('inf')
    
    def test_calmar_ratio(self):
        """Test calmar_ratio function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        
        # Test with provided max_drawdown
        assert calmar_ratio(returns, 0.15, 252) == pytest.approx(20.16, abs=1e-2)
        
        # Test without provided max_drawdown
        assert calmar_ratio(returns, None, 252) != float('inf')  # Just check it's finite
        
        # Zero drawdown
        assert calmar_ratio([0.01, 0.02, 0.03], 0, 252) == float('inf')
    
    def test_omega_ratio(self):
        """Test omega_ratio function."""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        
        # Basic test
        assert omega_ratio(returns, 0.005) == pytest.approx(3.3333333, abs=1e-6)
        
        # Higher threshold
        assert omega_ratio(returns, 0.015) == pytest.approx(0.5714286, abs=1e-6)
        
        # All returns above threshold
        assert omega_ratio([0.02, 0.03, 0.04], 0.01) == float('inf')
        
        # All returns below threshold
        assert omega_ratio([0.01, 0.005, 0.008], 0.02) == pytest.approx(0.0, abs=1e-10) 