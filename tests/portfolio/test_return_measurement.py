"""
Tests for the portfolio return measurement functions.
"""

import numpy as np
import pytest
from pypulate.portfolio.return_measurement import (
    simple_return, log_return, holding_period_return, annualized_return,
    time_weighted_return, money_weighted_return, arithmetic_return,
    geometric_return, total_return_index, dollar_weighted_return,
    modified_dietz_return, linked_modified_dietz_return, leveraged_return,
    market_neutral_return, beta_adjusted_return, long_short_equity_return
)


class TestReturnMeasurement:
    """Test class for portfolio return measurement functions."""
    
    def test_simple_return_scalar(self):
        """Test simple_return with scalar inputs."""
        assert simple_return(105, 100) == 0.05
        assert simple_return(90, 100) == -0.1
        assert simple_return(100, 100) == 0.0
        
    def test_simple_return_array(self):
        """Test simple_return with array inputs."""
        # Test with list inputs
        result = simple_return([105, 110, 108], [100, 100, 100])
        expected = np.array([0.05, 0.1, 0.08])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with numpy array inputs
        result = simple_return(np.array([105, 110]), np.array([100, 100]))
        expected = np.array([0.05, 0.1])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test mixed scalar and array
        result = simple_return(np.array([105, 110]), 100)
        expected = np.array([0.05, 0.1])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_log_return_scalar(self):
        """Test log_return with scalar inputs."""
        # Basic test
        assert log_return(105, 100) == pytest.approx(0.04879016)
        
        # Negative returns
        assert log_return(90, 100) == pytest.approx(-0.10536052)
        
        # Zero return
        assert log_return(100, 100) == 0.0
    
    def test_log_return_array(self):
        """Test log_return with array inputs."""
        # Test with list inputs
        result = log_return([105, 110, 108], [100, 100, 100])
        expected = np.array([0.04879016, 0.09531018, 0.07696104])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with numpy array inputs
        result = log_return(np.array([105, 110]), np.array([100, 100]))
        expected = np.array([0.04879016, 0.09531018])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_holding_period_return(self):
        """Test holding_period_return function."""
        # Basic test without dividends
        assert holding_period_return([100, 102, 105, 103, 106]) == 0.06
        
        # With dividends
        assert holding_period_return([100, 102, 105, 103, 106], [0, 1, 0, 2, 0]) == 0.09
        
        # Zero return with dividends
        assert holding_period_return([100, 100, 100], [1, 1, 1]) == 0.03
        
        # Negative return with dividends
        assert holding_period_return([100, 95, 90], [1, 1, 1]) == -0.07
    
    def test_annualized_return_scalar(self):
        """Test annualized_return with scalar inputs."""
        # Basic test
        assert annualized_return(0.2, 2) == pytest.approx(0.09544512)
        
        # One-year period
        assert annualized_return(0.1, 1) == pytest.approx(0.1)
        
        # Fractional year
        assert annualized_return(0.05, 0.5) == pytest.approx(0.10250000)
        
        # Negative return
        assert annualized_return(-0.1, 2) == pytest.approx(-0.05131670)
    
    def test_annualized_return_array(self):
        """Test annualized_return with array inputs."""
        # Test with list inputs
        result = annualized_return([0.2, 0.3, 0.15], [2, 3, 1.5])
        expected = np.array([0.09544512, 0.09139288, 0.0976534])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with numpy array inputs and scalar second argument
        result = annualized_return(np.array([0.4, 0.5]), 2)
        expected = np.array([0.18321596, 0.22474487])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_time_weighted_return(self):
        """Test time_weighted_return function."""
        # Basic test
        assert time_weighted_return([0.05, -0.02, 0.03, 0.04]) == pytest.approx(0.10226480)
        
        # Single period
        assert time_weighted_return([0.1]) == pytest.approx(0.1)
        
        # Zero return periods
        assert time_weighted_return([0, 0, 0]) == 0.0
        
        # Mix of positive and negative
        assert time_weighted_return([0.1, -0.1, 0.2]) == pytest.approx(0.18800000)
    
    def test_money_weighted_return(self):
        """Test money_weighted_return function."""
        # Basic test
        assert money_weighted_return([-1000, -500, 1700], [0, 0.5, 1], 0) == pytest.approx(0.16120410, abs=1e-5)
        
        # Simple case with initial value
        assert money_weighted_return([-100], [0.5], 110, 100) == pytest.approx(0.20000000, abs=1e-5)
    
    def test_arithmetic_return(self):
        """Test arithmetic_return function."""
        # Basic test
        assert arithmetic_return([100, 105, 103, 108, 110]) == pytest.approx(0.024503647)
        
        # Negative overall return
        assert arithmetic_return([100, 98, 96, 94, 92]) == pytest.approx(-0.020629523)
        
        # Mix of positive and negative
        assert arithmetic_return([100, 105, 95, 105, 95]) == pytest.approx(-0.008803258)
    
    def test_geometric_return(self):
        """Test geometric_return function."""
        # Basic test
        assert geometric_return([100, 105, 103, 108, 110]) == pytest.approx(0.024113689)
        
        # Negative overall return
        assert geometric_return([100, 98, 96, 94, 92]) == pytest.approx(-0.020629639)
        
        # Mix of positive and negative
        assert geometric_return([100, 105, 95, 105, 95]) == pytest.approx(-0.012741455)
    
    def test_total_return_index(self):
        """Test total_return_index function."""
        # Basic test without dividends
        result = total_return_index([100, 102, 105, 103, 106])
        expected = np.array([100, 102, 105, 103, 106])
        np.testing.assert_array_almost_equal(result, expected)
        
        # With dividends
        result = total_return_index([100, 102, 105, 103, 106], [0, 1, 0, 2, 0])
        expected = np.array([100, 103, 106.02941176, 106.02941176, 109.11764706])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with different array lengths
        with pytest.raises(ValueError):
            total_return_index([100, 102, 105], [0, 1])
    
    def test_dollar_weighted_return(self):
        """Test dollar_weighted_return function."""
        # Basic test
        assert dollar_weighted_return([-1000, -500, 200], [0, 30, 60], 1400) == pytest.approx(0.36174448, abs=1e-5)
    
    def test_modified_dietz_return(self):
        """Test modified_dietz_return function."""
        # Basic test
        assert modified_dietz_return(1000, 1200, [100, -50], [10, 20], 30) == pytest.approx(0.14285714)
        
        # No cash flows
        assert modified_dietz_return(1000, 1100, [], [], 30) == 0.1
    
    def test_linked_modified_dietz_return(self):
        """Test linked_modified_dietz_return function."""
        # Basic test
        assert linked_modified_dietz_return([0.05, -0.02, 0.03, 0.04]) == pytest.approx(0.10226480)
        
        # Single period
        assert linked_modified_dietz_return([0.1]) == pytest.approx(0.1)
    
    def test_leveraged_return_scalar(self):
        """Test leveraged_return with scalar inputs."""
        # Basic test
        assert leveraged_return(0.10, 2.0, 0.05) == pytest.approx(0.15)
        
        # No leverage
        assert leveraged_return(0.10, 1.0, 0.05) == pytest.approx(0.10)
        
        # High leverage
        assert leveraged_return(0.10, 3.0, 0.05) == pytest.approx(0.20)
        
        # Negative return
        assert leveraged_return(-0.05, 2.0, 0.05) == pytest.approx(-0.15)
    
    def test_leveraged_return_array(self):
        """Test leveraged_return with array inputs."""
        # Test with list inputs
        result = leveraged_return([0.10, 0.15], [2.0, 1.5], 0.05)
        expected = np.array([0.15, 0.2])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with mixed types
        result = leveraged_return(0.10, [2.0, 3.0], [0.05, 0.06])
        expected = np.array([0.15, 0.18])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_market_neutral_return_scalar(self):
        """Test market_neutral_return with scalar inputs."""
        # Basic test
        assert market_neutral_return(0.08, -0.05, 0.6, 0.4, 0.01) == 0.064
        
        # Equal weights
        assert market_neutral_return(0.10, -0.05, 0.5, 0.5, 0.01) == 0.07
        
        # No short borrowing cost
        assert market_neutral_return(0.08, -0.05, 0.6, 0.4) == 0.068
    
    def test_market_neutral_return_array(self):
        """Test market_neutral_return with array inputs."""
        # Test with list inputs
        result = market_neutral_return([0.08, 0.10], [-0.05, -0.03], 0.6, 0.4, 0.01)
        expected = np.array([0.064, 0.068])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with mixed types
        result = market_neutral_return(0.08, -0.05, [0.6, 0.7], [0.4, 0.3], [0.01, 0.02])
        expected = np.array([0.064, 0.065])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_beta_adjusted_return_scalar(self):
        """Test beta_adjusted_return with scalar inputs."""
        # Beta of 1.0
        assert beta_adjusted_return(0.10, 0.10, 1.0) == pytest.approx(0.0)
        
        # Beta greater than 1.0
        assert beta_adjusted_return(0.12, 0.10, 1.2) == pytest.approx(0.0)
        
        # Beta less than 1.0
        assert beta_adjusted_return(0.08, 0.10, 0.8) == pytest.approx(0.0)
        
        # Positive alpha
        assert beta_adjusted_return(0.15, 0.10, 1.0) == pytest.approx(0.05)
    
    def test_beta_adjusted_return_array(self):
        """Test beta_adjusted_return with array inputs."""
        # Test with list inputs
        result = beta_adjusted_return([0.12, 0.15], [0.10, 0.08], 1.2)
        expected = np.array([0.0, 0.054])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with numpy array inputs
        result = beta_adjusted_return(0.12, 0.10, [1.2, 1.5])
        expected = np.array([0.0, -0.03])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_long_short_equity_return_scalar(self):
        """Test long_short_equity_return with scalar inputs."""
        # Basic test
        assert long_short_equity_return(0.10, -0.05, 1.0, 0.5, 0.02, 0.01) == pytest.approx(0.14)
        
        # No short position
        assert long_short_equity_return(0.10, 0.0, 1.0, 0.0, 0.02, 0.01) == pytest.approx(0.1)
        
        # No long position
        assert long_short_equity_return(0.0, -0.05, 0.0, 0.5, 0.02, 0.01) == pytest.approx(0.06)
    
    def test_long_short_equity_return_array(self):
        """Test long_short_equity_return with array inputs."""
        # Test with list inputs
        result = long_short_equity_return([0.10, 0.12], [-0.05, -0.03], 1.0, 0.5, 0.02, 0.01)
        expected = np.array([0.14, 0.15])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with mixed types
        result = long_short_equity_return(0.10, -0.05, [1.0, 0.8], [0.5, 0.4], [0.02, 0.03], 0.01)
        expected = np.array([0.14, 0.122])
        np.testing.assert_array_almost_equal(result, expected) 