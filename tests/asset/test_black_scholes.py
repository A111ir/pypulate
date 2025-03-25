"""
Tests for the Black-Scholes option pricing model.
"""

import pytest
import numpy as np
from pypulate.asset import black_scholes, implied_volatility


def test_call_option_basic():
    """Test basic call option pricing."""
    result = black_scholes(
        option_type='call',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2
    )
    assert isinstance(result['price'], float)
    assert result['price'] > 0
    assert 0 <= result['delta'] <= 1  # Delta should be between 0 and 1 for calls
    assert result['gamma'] >= 0  # Gamma should be non-negative
    assert result['vega'] >= 0  # Vega should be non-negative


def test_put_option_basic():
    """Test basic put option pricing."""
    result = black_scholes(
        option_type='put',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2
    )
    assert isinstance(result['price'], float)
    assert result['price'] > 0
    assert -1 <= result['delta'] <= 0  # Delta should be between -1 and 0 for puts
    assert result['gamma'] >= 0  # Gamma should be non-negative
    assert result['vega'] >= 0  # Vega should be non-negative


def test_put_call_parity():
    """Test put-call parity relationship."""
    params = {
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2
    }
    
    call = black_scholes(option_type='call', **params)
    put = black_scholes(option_type='put', **params)
    
    # Put-call parity: C - P = S - K*exp(-rT)
    S = params['underlying_price']
    K = params['strike_price']
    r = params['risk_free_rate']
    T = params['time_to_expiry']
    
    parity_diff = abs((call['price'] - put['price']) - 
                     (S - K * np.exp(-r * T)))
    assert parity_diff < 1e-10  # Should be very precise


def test_input_validation():
    """Test input validation."""
    with pytest.raises(ValueError):
        black_scholes(
            option_type='invalid',  # Invalid option type
            underlying_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2
        )
    
    with pytest.raises(ValueError):
        black_scholes(
            option_type='call',
            underlying_price=-100.0,  # Invalid negative price
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2
        )
    
    with pytest.raises(ValueError):
        black_scholes(
            option_type='call',
            underlying_price=100.0,
            strike_price=100.0,
            time_to_expiry=-1.0,  # Invalid negative time
            risk_free_rate=0.05,
            volatility=0.2
        )


def test_dividend_effect():
    """Test the effect of dividends on option prices."""
    params = {
        'option_type': 'call',
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2
    }
    
    # Price without dividend
    price_no_div = black_scholes(**params)['price']
    
    # Price with dividend
    price_with_div = black_scholes(**params, dividend_yield=0.03)['price']
    
    # Call option price should decrease with dividends
    assert price_with_div < price_no_div


def test_volatility_effect():
    """Test the effect of volatility on option prices."""
    params = {
        'option_type': 'call',
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05
    }
    
    # Price with low volatility
    price_low_vol = black_scholes(**params, volatility=0.1)['price']
    
    # Price with high volatility
    price_high_vol = black_scholes(**params, volatility=0.3)['price']
    
    # Option price should increase with volatility
    assert price_high_vol > price_low_vol


def test_moneyness_effect():
    """Test the effect of moneyness on option prices."""
    base_params = {
        'option_type': 'call',
        'underlying_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2
    }
    
    # In-the-money call
    itm_call = black_scholes(**base_params, strike_price=90.0)
    # At-the-money call
    atm_call = black_scholes(**base_params, strike_price=100.0)
    # Out-of-the-money call
    otm_call = black_scholes(**base_params, strike_price=110.0)
    
    # Check price relationships
    assert itm_call['price'] > atm_call['price'] > otm_call['price']
    # Check delta relationships
    assert itm_call['delta'] > atm_call['delta'] > otm_call['delta']


def test_time_decay():
    """Test the effect of time decay (theta)."""
    result = black_scholes(
        option_type='call',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    # Theta should be negative for both calls and puts (time decay)
    assert result['theta'] < 0


def test_numerical_stability():
    """Test numerical stability with extreme parameters."""
    result = black_scholes(
        option_type='call',
        underlying_price=1000000.0,  # Very high price
        strike_price=1000000.0,
        time_to_expiry=10.0,  # Long time to expiry
        risk_free_rate=0.1,
        volatility=0.5  # High volatility
    )
    assert np.isfinite(result['price'])
    assert np.isfinite(result['delta'])
    assert np.isfinite(result['gamma'])
    assert np.isfinite(result['theta'])
    assert np.isfinite(result['vega'])
    assert np.isfinite(result['rho'])


def test_implied_volatility_basic():
    """Test basic implied volatility calculation."""
    # First calculate a Black-Scholes price
    true_vol = 0.2
    bs_result = black_scholes(
        option_type='call',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=true_vol
    )
    
    # Then use that price to recover the volatility
    implied_vol_result = implied_volatility(
        option_type='call',
        market_price=bs_result['price'],
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05
    )
    
    assert abs(implied_vol_result['implied_volatility'] - true_vol) < 1e-4


def test_implied_volatility_bounds():
    """Test implied volatility calculation with extreme market prices."""
    params = {
        'option_type': 'call',
        'underlying_price': 100.0,
        'strike_price': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05
    }
    
    high_price_result = implied_volatility(
        market_price=25.0,  
        **params
    )
    assert 'error' not in high_price_result
    assert high_price_result['implied_volatility'] > 0
    
    low_price_result = implied_volatility(
        market_price=5.0,  
        **params
    )
    assert 'error' not in low_price_result
    assert low_price_result['implied_volatility'] > 0


def test_implied_volatility_error_cases():
    """Test error cases for implied volatility calculation."""
    with pytest.raises(ValueError):
        implied_volatility(
            option_type='invalid', 
            market_price=10.0,
            underlying_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05
        )
    
    with pytest.raises(ValueError):
        implied_volatility(
            option_type='call',
            market_price=-10.0,  
            underlying_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05
        ) 