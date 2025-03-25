"""
Tests for the binomial tree option pricing model.
"""

import pytest
import numpy as np
from pypulate.asset import binomial_tree


def test_european_call_basic():
    """Test basic European call option pricing."""
    result = binomial_tree(
        option_type='european_call',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        steps=100
    )
    assert isinstance(result['price'], float)
    assert result['price'] > 0
    assert 0 <= result['delta'] <= 1  # Delta should be between 0 and 1 for calls
    assert result['gamma'] >= 0  # Gamma should be non-negative


def test_european_put_basic():
    """Test basic European put option pricing."""
    result = binomial_tree(
        option_type='european_put',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        steps=100
    )
    assert isinstance(result['price'], float)
    assert result['price'] > 0
    assert -1 <= result['delta'] <= 0  # Delta should be between -1 and 0 for puts
    assert result['gamma'] >= 0  # Gamma should be non-negative


def test_american_put_early_exercise():
    """Test American put option with optimal early exercise."""
    result = binomial_tree(
        option_type='american_put',
        underlying_price=100.0,
        strike_price=110.0,
        time_to_expiry=1.0,
        risk_free_rate=0.01,  # Low interest rate
        volatility=0.2,
        steps=100,
        dividend_yield=0.05  # High dividend yield
    )
    assert result['price'] > 0
    assert result['early_exercise_optimal'] is not None


def test_american_vs_european_put():
    """Test that American put is worth at least as much as European put."""
    params = {
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'steps': 100
    }
    
    american = binomial_tree(option_type='american_put', **params)
    european = binomial_tree(option_type='european_put', **params)
    
    assert american['price'] >= european['price']


def test_put_call_parity():
    """Test put-call parity relationship for European options."""
    params = {
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'steps': 100
    }
    
    call = binomial_tree(option_type='european_call', **params)
    put = binomial_tree(option_type='european_put', **params)
    
    # Put-call parity: C - P = S - K*exp(-rT)
    S = params['underlying_price']
    K = params['strike_price']
    r = params['risk_free_rate']
    T = params['time_to_expiry']
    
    parity_diff = abs((call['price'] - put['price']) - 
                     (S - K * np.exp(-r * T)))
    assert parity_diff < 1e-2  # Allow for small numerical differences


def test_convergence():
    """Test convergence with increasing number of steps."""
    base_params = {
        'option_type': 'european_call',
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2
    }
    
    # Test with increasing number of steps
    price_50 = binomial_tree(**base_params, steps=50)['price']
    price_100 = binomial_tree(**base_params, steps=100)['price']
    price_200 = binomial_tree(**base_params, steps=200)['price']
    
    # Prices should converge
    diff_50_100 = abs(price_100 - price_50)
    diff_100_200 = abs(price_200 - price_100)
    assert diff_100_200 < diff_50_100


def test_input_validation():
    """Test input validation."""
    with pytest.raises(ValueError):
        binomial_tree(
            option_type='invalid_type',
            underlying_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2
        )
    
    with pytest.raises(ValueError):
        binomial_tree(
            option_type='european_call',
            underlying_price=-100.0,  # Invalid negative price
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2
        )
    
    with pytest.raises(ValueError):
        binomial_tree(
            option_type='european_call',
            underlying_price=100.0,
            strike_price=100.0,
            time_to_expiry=-1.0,  # Invalid negative time
            risk_free_rate=0.05,
            volatility=0.2
        )


def test_dividend_effect():
    """Test the effect of dividends on option prices."""
    base_params = {
        'option_type': 'european_call',
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'steps': 100
    }
    
    # Price without dividend
    price_no_div = binomial_tree(**base_params)['price']
    
    # Price with dividend
    price_with_div = binomial_tree(**base_params, dividend_yield=0.03)['price']
    
    # Call option price should decrease with dividends
    assert price_with_div < price_no_div


def test_volatility_effect():
    """Test the effect of volatility on option prices."""
    base_params = {
        'option_type': 'european_call',
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'steps': 100
    }
    
    # Price with low volatility
    price_low_vol = binomial_tree(**base_params, volatility=0.1)['price']
    
    # Price with high volatility
    price_high_vol = binomial_tree(**base_params, volatility=0.3)['price']
    
    # Option price should increase with volatility
    assert price_high_vol > price_low_vol


def test_numerical_stability():
    """Test numerical stability with extreme parameters."""
    result = binomial_tree(
        option_type='european_call',
        underlying_price=1000000.0,  # Very high price
        strike_price=1000000.0,
        time_to_expiry=10.0,  # Long time to expiry
        risk_free_rate=0.1,
        volatility=0.5,  # High volatility
        steps=200
    )
    assert np.isfinite(result['price'])
    assert np.isfinite(result['delta'])
    assert np.isfinite(result['gamma']) 