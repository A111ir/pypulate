"""Tests for return measurement functions."""

import pytest
import numpy as np
from pypulate.portfolio.return_measurement import (
    simple_return,
    log_return,
    holding_period_return,
    annualized_return,
    time_weighted_return,
    money_weighted_return,
    arithmetic_return,
    geometric_return,
    total_return_index,
    dollar_weighted_return,
    modified_dietz_return,
    linked_modified_dietz_return,
    leveraged_return,
    market_neutral_return,
    beta_adjusted_return,
    long_short_equity_return
)


def test_simple_return():
    """Test simple return calculation."""
    # Test single values
    assert simple_return(105, 100) == pytest.approx(0.05)
    assert simple_return(95, 100) == pytest.approx(-0.05)
    
    # Test lists
    returns = simple_return([105, 110, 108], [100, 100, 100])
    np.testing.assert_array_almost_equal(returns, [0.05, 0.10, 0.08])
    
    # Test numpy arrays
    returns = simple_return(np.array([105, 110]), np.array([100, 100]))
    np.testing.assert_array_almost_equal(returns, [0.05, 0.10])


def test_log_return():
    """Test logarithmic return calculation."""
    # Test single values
    assert log_return(105, 100) == pytest.approx(0.04879016)
    assert log_return(95, 100) == pytest.approx(-0.05129329)
    
    # Test lists
    returns = log_return([105, 110, 108], [100, 100, 100])
    np.testing.assert_array_almost_equal(
        returns, [0.04879016, 0.09531018, 0.07696104]
    )
    
    # Test numpy arrays
    returns = log_return(np.array([105, 110]), np.array([100, 100]))
    np.testing.assert_array_almost_equal(returns, [0.04879016, 0.09531018])


def test_holding_period_return():
    """Test holding period return calculation."""
    # Test without dividends
    assert holding_period_return([100, 102, 105, 103, 106]) == pytest.approx(0.06)
    
    # Test with dividends
    assert holding_period_return(
        [100, 102, 105, 103, 106], 
        [0, 1, 0, 2, 0]
    ) == pytest.approx(0.09)


def test_annualized_return():
    """Test annualized return calculation."""
    # Test single values
    result = annualized_return(0.2, 2)
    assert result == pytest.approx(0.0954, abs=1e-4)
    
    result = annualized_return(0.4, 2)
    assert result == pytest.approx(0.1832, abs=1e-4)
    
    # Test lists
    returns = annualized_return([0.2, 0.3, 0.15], [2, 3, 1.5])
    expected = [0.0954, 0.0914, 0.0977]
    np.testing.assert_array_almost_equal(returns, expected, decimal=4)


def test_time_weighted_return():
    """Test time-weighted return calculation."""
    returns = [0.05, -0.02, 0.03, 0.04]
    result = time_weighted_return(returns)
    assert result == pytest.approx(0.1023, abs=1e-4)
    
    # Test with different return patterns
    returns = [-0.05, 0.05]  # Should approximately net to zero
    result = time_weighted_return(returns)
    assert result == pytest.approx(-0.0025, abs=1e-4)


def test_money_weighted_return():
    """Test money-weighted return (IRR) calculation."""
    cash_flows = [-1000, -500, 1700]
    times = [0, 0.5, 1]
    final_value = 0
    
    irr = money_weighted_return(cash_flows, times, final_value)
    assert irr == pytest.approx(0.1612, abs=1e-4)


def test_arithmetic_return():
    """Test arithmetic average return calculation."""
    prices = [100, 105, 103, 108, 110]
    result = arithmetic_return(prices)
    assert result == pytest.approx(0.0245, abs=1e-4)


def test_geometric_return():
    """Test geometric average return calculation."""
    prices = [100, 105, 103, 108, 110]
    result = geometric_return(prices)
    assert result == pytest.approx(0.0241, abs=1e-4)


def test_total_return_index():
    """Test total return index calculation."""
    # Test without dividends
    prices = [100, 102, 105, 103, 106]
    tri = total_return_index(prices)
    expected = np.array([100., 102., 105., 103., 106.])
    np.testing.assert_array_almost_equal(tri, expected, decimal=4)
    
    # Test with dividends
    prices = [100, 102, 105, 103, 106]
    dividends = [0, 1, 0, 2, 0]
    tri = total_return_index(prices, dividends)
    expected = np.array([100., 103., 106.03, 106.03, 109.12])
    np.testing.assert_array_almost_equal(tri, expected, decimal=2)


def test_dollar_weighted_return():
    """Test dollar-weighted return calculation."""
    cash_flows = [-1000, -500, 200]
    dates = [0, 30, 60]
    end_value = 1400
    
    return_value = dollar_weighted_return(cash_flows, dates, end_value)
    assert return_value == pytest.approx(0.3617, abs=1e-4)


def test_modified_dietz_return():
    """Test Modified Dietz return calculation."""
    start_value = 1000
    end_value = 1200
    cash_flows = [100, -50]
    flow_days = [10, 20]
    total_days = 30
    
    return_value = modified_dietz_return(
        start_value, end_value, cash_flows, flow_days, total_days
    )
    assert return_value == pytest.approx(0.1429, abs=1e-4)


def test_linked_modified_dietz_return():
    """Test linked Modified Dietz return calculation."""
    returns = [0.05, -0.02, 0.03, 0.04]
    result = linked_modified_dietz_return(returns)
    assert result == pytest.approx(0.1023, abs=1e-4)


def test_market_neutral_return():
    """Test market-neutral return calculation."""
    # Test single values
    result = market_neutral_return(0.08, -0.05, 0.6, 0.4, 0.01)
    assert result == pytest.approx(0.064, abs=1e-4)
    
    # Test arrays
    returns = market_neutral_return(
        [0.08, 0.10], [-0.05, -0.03], 0.6, 0.4, 0.01
    )
    expected = [0.064, 0.068]
    np.testing.assert_array_almost_equal(returns, expected, decimal=4)


def test_beta_adjusted_return():
    """Test beta-adjusted return calculation."""
    # Test single values
    result = beta_adjusted_return(0.12, 0.10, 1.2)
    assert result == pytest.approx(0.0, abs=1e-4)
    
    # Test arrays
    returns = beta_adjusted_return([0.12, 0.15], [0.10, 0.08], 1.2)
    expected = [0.0, 0.054]
    np.testing.assert_array_almost_equal(returns, expected, decimal=4)


def test_long_short_equity_return():
    """Test long-short equity return calculation."""
    # Test single values
    result = long_short_equity_return(
        0.10, -0.05, 1.0, 0.5, 0.02, 0.01
    )
    assert result == pytest.approx(0.14, abs=1e-4)
    
    # Test arrays
    returns = long_short_equity_return(
        [0.10, 0.12], [-0.05, -0.03], 1.0, 0.5, 0.02, 0.01
    )
    expected = [0.14, 0.15]
    np.testing.assert_array_almost_equal(returns, expected, decimal=4)
