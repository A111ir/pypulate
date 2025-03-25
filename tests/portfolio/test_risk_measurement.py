"""
Tests for the portfolio risk measurement functions.
"""

import numpy as np
import pytest
from pypulate.portfolio.risk_measurement import (
    standard_deviation, semi_standard_deviation, tracking_error, capm_beta,
    value_at_risk, covariance_matrix, correlation_matrix, conditional_value_at_risk,
    drawdown
)


class TestRiskMeasurement:
    """Test class for portfolio risk measurement functions."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns for testing."""
        return [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.015, 0.008]
    
    @pytest.fixture
    def sample_benchmark_returns(self):
        """Create sample benchmark returns for testing."""
        return [0.005, 0.015, -0.005, 0.02, 0.008, -0.01, 0.01, 0.005]

    @pytest.fixture
    def sample_multi_asset_returns(self):
        """Create sample multi-asset returns matrix for testing."""
        return [
            [0.01, 0.005, 0.003],   # Day 1
            [0.02, 0.01, -0.01],    # Day 2
            [-0.01, -0.005, 0.002], # Day 3
            [0.03, 0.02, 0.01],     # Day 4
            [0.01, 0.01, 0.005]     # Day 5
        ]
    
    def test_standard_deviation(self, sample_returns):
        """Test standard_deviation function."""
        # Basic test
        std = standard_deviation(sample_returns)
        assert isinstance(std, float)
        assert std == pytest.approx(0.0160039, abs=1e-6)
        
        # Annualized
        annualized_std = standard_deviation(sample_returns, annualize=True)
        assert annualized_std == pytest.approx(0.254054, abs=1e-5)
        
        # Custom periods per year
        monthly_annualized_std = standard_deviation(sample_returns, annualize=True, periods_per_year=12)
        assert monthly_annualized_std == pytest.approx(0.0554392, abs=1e-6)
        
        # NumPy array input
        np_std = standard_deviation(np.array(sample_returns))
        assert np_std == pytest.approx(std, abs=1e-10)
    
    def test_semi_standard_deviation(self, sample_returns):
        """Test semi_standard_deviation function."""
        # Basic test with zero threshold
        semi_std = semi_standard_deviation(sample_returns)
        assert isinstance(semi_std, float)
        assert semi_std == pytest.approx(0.0070711, abs=1e-6)
        
        # With custom threshold
        semi_std_high = semi_standard_deviation(sample_returns, threshold=0.01)
        assert semi_std_high > semi_std  # Higher threshold means more returns below it
        
        # Annualized
        annualized_semi_std = semi_standard_deviation(sample_returns, annualize=True)
        assert annualized_semi_std == pytest.approx(0.11225, abs=1e-5)
        
        # No returns below threshold
        semi_std_all_above = semi_standard_deviation([0.02, 0.03, 0.04], threshold=0.01)
        assert semi_std_all_above == 0.0
    
    def test_tracking_error(self, sample_returns, sample_benchmark_returns):
        """Test tracking_error function."""
        # Basic test
        te = tracking_error(sample_returns, sample_benchmark_returns)
        assert isinstance(te, float)
        assert te == pytest.approx(0.006379, abs=1e-6)
        
        # Annualized
        annualized_te = tracking_error(sample_returns, sample_benchmark_returns, annualize=True)
        assert annualized_te == pytest.approx(0.10127, abs=1e-5)
        
        # Same returns should give zero tracking error
        zero_te = tracking_error(sample_returns, sample_returns)
        assert zero_te == 0.0
        
        # Different length inputs should raise an error
        with pytest.raises(ValueError):
            tracking_error(sample_returns, sample_benchmark_returns[:-1])
    
    def test_capm_beta(self, sample_returns, sample_benchmark_returns):
        """Test capm_beta function."""
        # Basic test
        beta = capm_beta(sample_returns, sample_benchmark_returns)
        assert isinstance(beta, float)
        assert beta == pytest.approx(1.6154, abs=1e-4)
        
        # Same returns should give beta of 1.0
        same_beta = capm_beta(sample_returns, sample_returns)
        assert same_beta == 1.0
        
        # Different length inputs should raise an error
        with pytest.raises(ValueError):
            capm_beta(sample_returns, sample_benchmark_returns[:-1])
        
        # NumPy array input
        np_beta = capm_beta(np.array(sample_returns), np.array(sample_benchmark_returns))
        assert np_beta == pytest.approx(beta, abs=1e-10)
    
    def test_value_at_risk(self, sample_returns):
        """Test value_at_risk function."""
        # Historical method
        historical_var = value_at_risk(sample_returns, confidence_level=0.95, method='historical')
        assert isinstance(historical_var, float)
        assert historical_var > 0.0  # VaR should be positive
        
        # Parametric method
        parametric_var = value_at_risk(sample_returns, confidence_level=0.95, method='parametric')
        assert isinstance(parametric_var, float)
        assert parametric_var > 0.0  # VaR should be positive
        
        # Monte Carlo method
        monte_carlo_var = value_at_risk(sample_returns, confidence_level=0.95, method='monte_carlo')
        assert isinstance(monte_carlo_var, float)
        assert monte_carlo_var > 0.0  # VaR should be positive
        
        # With custom current value
        scaled_var = value_at_risk(sample_returns, confidence_level=0.95, current_value=1000000)
        assert scaled_var == pytest.approx(historical_var * 1000000, abs=1)
        
        # Invalid method should raise an error
        with pytest.raises(ValueError):
            value_at_risk(sample_returns, method='invalid')
    
    def test_covariance_matrix(self, sample_multi_asset_returns):
        """Test covariance_matrix function."""
        # Basic test
        cov_matrix = covariance_matrix(sample_multi_asset_returns)
        assert isinstance(cov_matrix, np.ndarray)
        assert cov_matrix.shape == (3, 3)  # 3x3 for 3 assets
        
        # Check properties of covariance matrix
        assert np.all(np.diag(cov_matrix) >= 0)  # Diagonal elements (variances) should be non-negative
        assert np.allclose(cov_matrix, cov_matrix.T)  # Should be symmetric
        
        # Test with NumPy array input
        np_cov_matrix = covariance_matrix(np.array(sample_multi_asset_returns))
        np.testing.assert_array_almost_equal(np_cov_matrix, cov_matrix)
    
    def test_correlation_matrix(self, sample_multi_asset_returns):
        """Test correlation_matrix function."""
        # Basic test
        corr_matrix = correlation_matrix(sample_multi_asset_returns)
        assert isinstance(corr_matrix, np.ndarray)
        assert corr_matrix.shape == (3, 3)  # 3x3 for 3 assets
        
        # Check properties of correlation matrix
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal elements should be 1
        assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)  # Values between -1 and 1
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric
        
        # Test with NumPy array input
        np_corr_matrix = correlation_matrix(np.array(sample_multi_asset_returns))
        np.testing.assert_array_almost_equal(np_corr_matrix, corr_matrix)
    
    def test_conditional_value_at_risk(self, sample_returns):
        """Test conditional_value_at_risk function."""
        # Historical method
        historical_cvar = conditional_value_at_risk(sample_returns, confidence_level=0.95, method='historical')
        assert isinstance(historical_cvar, float)
        assert historical_cvar > 0.0  # CVaR should be positive
        
        # Parametric method
        parametric_cvar = conditional_value_at_risk(sample_returns, confidence_level=0.95, method='parametric')
        assert isinstance(parametric_cvar, float)
        assert parametric_cvar > 0.0  # CVaR should be positive
        
        # CVaR should be greater than or equal to VaR
        historical_var = value_at_risk(sample_returns, confidence_level=0.95, method='historical')
        assert historical_cvar >= historical_var
        
        # With custom current value
        scaled_cvar = conditional_value_at_risk(sample_returns, confidence_level=0.95, current_value=1000000)
        assert scaled_cvar == pytest.approx(historical_cvar * 1000000, abs=1)
        
        # Invalid method should raise an error
        with pytest.raises(ValueError):
            conditional_value_at_risk(sample_returns, method='invalid')
    
    def test_drawdown(self, sample_returns):
        """Test drawdown function."""
        # Basic test as numpy array
        drawdowns, max_dd, start_idx, end_idx = drawdown(sample_returns)
        assert isinstance(drawdowns, np.ndarray)
        assert isinstance(max_dd, float)
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        
        assert max_dd > 0.0  # Max drawdown should be positive
        assert 0 <= start_idx < len(sample_returns)
        assert 0 <= end_idx < len(sample_returns)
        assert start_idx <= end_idx
        
        # Test as list
        drawdowns_list, max_dd_list, start_idx_list, end_idx_list = drawdown(sample_returns, as_list=True)
        assert isinstance(drawdowns_list, list)
        assert max_dd == max_dd_list
        assert start_idx == start_idx_list
        assert end_idx == end_idx_list
        
        # Test with only rising prices (no drawdown)
        rising_returns = [0.01, 0.02, 0.03]
        _, rising_max_dd, _, _ = drawdown(rising_returns)
        assert rising_max_dd == 0.0
        
        # Test with numpy array input
        np_drawdowns, np_max_dd, np_start_idx, np_end_idx = drawdown(np.array(sample_returns))
        np.testing.assert_array_almost_equal(np_drawdowns, drawdowns)
        assert np_max_dd == pytest.approx(max_dd, abs=1e-10)
        assert np_start_idx == start_idx
        assert np_end_idx == end_idx 