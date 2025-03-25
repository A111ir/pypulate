import pytest
import numpy as np
from pypulate.asset.monte_carlo import ( #type: ignore
    monte_carlo_option_pricing,
    price_action_monte_carlo,
    hybrid_price_action_monte_carlo
)


def test_basic_monte_carlo():
    """Test basic Monte Carlo option pricing."""
    result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=100,
        strike_price=100,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        simulations=10000,
        seed=42
    )
    
    assert isinstance(result, dict)
    assert 8.0 <= result['price'] <= 12.0  # Reasonable range for ATM call
    assert result['standard_error'] > 0
    assert result['confidence_interval'][0] < result['price'] < result['confidence_interval'][1]


def test_put_call_parity():
    """Test put-call parity relationship."""
    params = {
        'underlying_price': 100,
        'strike_price': 100,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'simulations': 50000,
        'seed': 42
    }
    
    call_result = monte_carlo_option_pricing(option_type='european_call', **params)
    put_result = monte_carlo_option_pricing(option_type='european_put', **params)
    
    # Put-call parity: C - P = S - K*exp(-rT)
    S = params['underlying_price']
    K = params['strike_price']
    r = params['risk_free_rate']
    T = params['time_to_expiry']
    
    parity_diff = abs((call_result['price'] - put_result['price']) - 
                     (S - K * np.exp(-r * T)))
    
    assert parity_diff < 1.0  # Allow for Monte Carlo error


def test_asian_options():
    """Test Asian option pricing."""
    params = {
        'underlying_price': 100,
        'strike_price': 100,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'simulations': 10000,
        'seed': 42
    }
    
    asian_call = monte_carlo_option_pricing(option_type='asian_call', **params)
    asian_put = monte_carlo_option_pricing(option_type='asian_put', **params)
    euro_call = monte_carlo_option_pricing(option_type='european_call', **params)
    
    # Asian options should be cheaper than European options
    assert asian_call['price'] < euro_call['price']
    assert asian_call['price'] > 0
    assert asian_put['price'] > 0


def test_lookback_options():
    """Test lookback option pricing."""
    params = {
        'underlying_price': 100,
        'strike_price': 100,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'simulations': 10000,
        'seed': 42
    }
    
    lookback_call = monte_carlo_option_pricing(option_type='lookback_call', **params)
    lookback_put = monte_carlo_option_pricing(option_type='lookback_put', **params)
    euro_call = monte_carlo_option_pricing(option_type='european_call', **params)
    
    # Lookback options should be more expensive than European options
    assert lookback_call['price'] > euro_call['price']
    assert lookback_call['price'] > 0
    assert lookback_put['price'] > 0


def test_jump_diffusion():
    """Test jump diffusion model."""
    base_result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=100,
        strike_price=100,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        jump_intensity=0.0,
        simulations=10000,
        seed=42
    )
    
    jump_result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=100,
        strike_price=100,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        jump_intensity=2.0,
        jump_mean=-0.05,
        jump_std=0.1,
        simulations=10000,
        seed=42
    )
    
    # Price should be different with jumps
    assert abs(jump_result['price'] - base_result['price']) > 0.1
    assert jump_result['standard_error'] > base_result['standard_error']


def test_price_action():
    """Test price action Monte Carlo."""
    result = price_action_monte_carlo(
        option_type='european_call',
        underlying_price=100,
        strike_price=100,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        support_levels=[95, 97],
        resistance_levels=[103, 105],
        respect_level_strength=0.7,
        simulations=10000,
        seed=42
    )
    
    assert isinstance(result, dict)
    assert 8.0 <= result['price'] <= 12.0  # Reasonable range for ATM call
    assert result['standard_error'] > 0
    assert result['confidence_interval'][0] < result['price'] < result['confidence_interval'][1]


def test_hybrid_model():
    """Test hybrid price action Monte Carlo."""
    result = hybrid_price_action_monte_carlo(
        option_type='european_call',
        underlying_price=100,
        strike_price=100,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        support_levels=[95, 97],
        resistance_levels=[103, 105],
        mean_reversion_params={'long_term_mean': 100, 'mean_reversion_rate': 1.0},
        jump_params={'jump_intensity': 1.0, 'jump_mean': -0.05, 'jump_std': 0.1},
        price_action_weight=0.4,
        mean_reversion_weight=0.3,
        jump_diffusion_weight=0.3,
        simulations=10000,
        seed=42
    )
    
    assert isinstance(result, dict)
    assert 8.0 <= result['price'] <= 12.0  # Reasonable range for ATM call
    assert result['standard_error'] > 0
    assert result['confidence_interval'][0] < result['price'] < result['confidence_interval'][1]
    assert abs(result['price_action_price'] - result['mean_reversion_price']) > 0.01
    assert abs(result['price_action_price'] - result['jump_diffusion_price']) > 0.01


def test_input_validation():
    """Test input validation."""
    with pytest.raises(ValueError):
        monte_carlo_option_pricing(
            option_type='invalid_type',
            underlying_price=100,
            strike_price=100,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2
        )
    
    with pytest.raises(ValueError):
        monte_carlo_option_pricing(
            option_type='european_call',
            underlying_price=-100,  # Invalid negative price
            strike_price=100,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2
        )
    
    with pytest.raises(ValueError):
        price_action_monte_carlo(
            option_type='european_call',
            underlying_price=100,
            strike_price=100,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            support_levels=[95],
            resistance_levels=[105],
            respect_level_strength=1.5  # Invalid strength > 1
        )
    
    with pytest.raises(ValueError):
        hybrid_price_action_monte_carlo(
            option_type='european_call',
            underlying_price=100,
            strike_price=100,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            support_levels=[95],
            resistance_levels=[105],
            price_action_weight=0.5,
            mean_reversion_weight=0.3,
            jump_diffusion_weight=0.3  # Weights sum to 1.1
        )


def test_antithetic_variance_reduction():
    """Test that antithetic variates reduce variance."""
    params = {
        'option_type': 'european_call',
        'underlying_price': 100,
        'strike_price': 100,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'simulations': 10000,
        'seed': 42
    }
    
    # Run with and without antithetic variates
    result_with = monte_carlo_option_pricing(antithetic=True, **params)
    result_without = monte_carlo_option_pricing(antithetic=False, **params)
    
    # Antithetic variates should reduce standard error
    assert result_with['standard_error'] < result_without['standard_error']


def test_reproducibility():
    """Test that results are reproducible with same seed."""
    params = {
        'option_type': 'european_call',
        'underlying_price': 100,
        'strike_price': 100,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'simulations': 10000,
        'seed': 42
    }
    
    result1 = monte_carlo_option_pricing(**params)
    result2 = monte_carlo_option_pricing(**params)
    
    assert result1['price'] == result2['price']
    assert result1['standard_error'] == result2['standard_error'] 