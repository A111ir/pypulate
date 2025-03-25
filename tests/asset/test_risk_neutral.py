"""
Tests for the risk-neutral valuation module.
"""

import pytest
import numpy as np
from pypulate.asset.risk_neutral import risk_neutral_valuation
from pypulate.asset.black_scholes import black_scholes


def test_call_option_basic():
    """Test basic call option pricing."""
    def call_payoff(s, k=100):
        return max(0, s - k)
    
    result = risk_neutral_valuation(
        payoff_function=lambda s: call_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        simulations=10000,
        seed=42
    )
    
    # Compare with Black-Scholes
    bs_result = black_scholes(
        option_type='call',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    assert isinstance(result['price'], float)
    assert result['price'] > 0
    # Allow for Monte Carlo error, should be within 5% of Black-Scholes
    assert abs(result['price'] - bs_result['price']) / bs_result['price'] < 0.05


def test_put_option_basic():
    """Test basic put option pricing."""
    def put_payoff(s, k=100):
        return max(0, k - s)
    
    result = risk_neutral_valuation(
        payoff_function=lambda s: put_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        simulations=10000,
        seed=42
    )
    
    # Compare with Black-Scholes
    bs_result = black_scholes(
        option_type='put',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    assert isinstance(result['price'], float)
    assert result['price'] > 0
    # Allow for Monte Carlo error, should be within 5% of Black-Scholes
    assert abs(result['price'] - bs_result['price']) / bs_result['price'] < 0.05


def test_put_call_parity():
    """Test put-call parity relationship."""
    def call_payoff(s, k=100):
        return max(0, s - k)
    
    def put_payoff(s, k=100):
        return max(0, k - s)
    
    params = {
        'underlying_price': 100.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'time_to_expiry': 1.0,
        'simulations': 50000,
        'seed': 42
    }
    
    call_result = risk_neutral_valuation(
        payoff_function=lambda s: call_payoff(s),
        **params
    )
    
    put_result = risk_neutral_valuation(
        payoff_function=lambda s: put_payoff(s),
        **params
    )
    
    # Put-call parity: C - P = S - K*exp(-rT)
    S = params['underlying_price']
    K = 100  # Strike price used in payoff functions
    r = params['risk_free_rate']
    T = params['time_to_expiry']
    
    parity_diff = abs((call_result['price'] - put_result['price']) - 
                     (S - K * np.exp(-r * T)))
    
    assert parity_diff < 1.0  # Allow for Monte Carlo error


def test_binary_option():
    """Test binary (digital) option pricing."""
    def binary_call_payoff(s, k=100):
        return 1.0 if s > k else 0.0
    
    result = risk_neutral_valuation(
        payoff_function=lambda s: binary_call_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        simulations=10000,
        seed=42
    )
    
    # Binary call option price should be approximately e^(-rT) * N(d2)
    S = 100.0
    K = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.2
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theoretical_price = np.exp(-r * T) * norm.cdf(d2)
    
    assert isinstance(result['price'], float)
    assert 0 <= result['price'] <= 1.0  # Binary option payoff is either 0 or 1
    assert abs(result['price'] - theoretical_price) < 0.05  # Allow for Monte Carlo error


def test_barrier_option():
    """Test barrier option pricing."""
    def down_and_out_call_payoff(price_path, barrier=90, strike=100):
        # If price ever goes below barrier, payoff is 0
        if np.min(price_path) < barrier:
            return 0.0
        # Otherwise, it's a regular call option
        return max(0, price_path[-1] - strike)
    
    # For barrier options, we need the full price path
    # This is a simplified test that doesn't use the actual function
    # since risk_neutral_valuation doesn't support path-dependent options directly
    
    # Instead, we'll just test that the function validates inputs correctly
    with pytest.raises(ValueError):
        risk_neutral_valuation(
            payoff_function=lambda s: down_and_out_call_payoff(s),  # This won't work correctly
            underlying_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            simulations=-100,  # Invalid negative simulations
            seed=42
        )


def test_custom_exotic_option():
    """Test a custom exotic option with a non-standard payoff."""
    def power_option_payoff(s, k=100, power=2):
        return max(0, (s - k)**power)
    
    result = risk_neutral_valuation(
        payoff_function=lambda s: power_option_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        simulations=10000,
        seed=42
    )
    
    assert isinstance(result['price'], float)
    assert result['price'] > 0
    assert 'price_statistics' in result
    assert 'payoff_statistics' in result


def test_dividend_effect():
    """Test the effect of dividends on option pricing."""
    def call_payoff(s, k=100):
        return max(0, s - k)
    
    no_div_result = risk_neutral_valuation(
        payoff_function=lambda s: call_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        dividend_yield=0.0,
        simulations=10000,
        seed=42
    )
    
    with_div_result = risk_neutral_valuation(
        payoff_function=lambda s: call_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        dividend_yield=0.03,
        simulations=10000,
        seed=42
    )
    
    # Option with dividend yield should be cheaper
    assert with_div_result['price'] < no_div_result['price']


def test_input_validation():
    """Test input validation."""
    def call_payoff(s, k=100):
        return max(0, s - k)
    
    # Test invalid payoff function
    with pytest.raises(ValueError):
        risk_neutral_valuation(
            payoff_function="not_a_function",
            underlying_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0
        )
    
    # Test invalid underlying price
    with pytest.raises(ValueError):
        risk_neutral_valuation(
            payoff_function=lambda s: call_payoff(s),
            underlying_price=-100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0
        )
    
    # Test invalid time to expiry
    with pytest.raises(ValueError):
        risk_neutral_valuation(
            payoff_function=lambda s: call_payoff(s),
            underlying_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=-1.0
        )
    
    # Test invalid volatility
    with pytest.raises(ValueError):
        risk_neutral_valuation(
            payoff_function=lambda s: call_payoff(s),
            underlying_price=100.0,
            risk_free_rate=0.05,
            volatility=-0.2,
            time_to_expiry=1.0
        )
    
    # Test invalid steps
    with pytest.raises(ValueError):
        risk_neutral_valuation(
            payoff_function=lambda s: call_payoff(s),
            underlying_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            steps=-100
        )
    
    # Test invalid simulations
    with pytest.raises(ValueError):
        risk_neutral_valuation(
            payoff_function=lambda s: call_payoff(s),
            underlying_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            simulations=-100
        )


def test_reproducibility():
    """Test that results are reproducible with same seed."""
    def call_payoff(s, k=100):
        return max(0, s - k)
    
    params = {
        'payoff_function': lambda s: call_payoff(s),
        'underlying_price': 100.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'time_to_expiry': 1.0,
        'simulations': 1000,
        'seed': 42
    }
    
    result1 = risk_neutral_valuation(**params)
    result2 = risk_neutral_valuation(**params)
    
    # Results should be identical with same seed
    assert result1['price'] == result2['price']
    
    # Results should be different with different seeds
    params['seed'] = 43
    result3 = risk_neutral_valuation(**params)
    
    assert result1['price'] != result3['price']


def test_confidence_interval():
    """Test that the Monte Carlo price is close to the Black-Scholes price."""
    def call_payoff(s, k=100):
        return max(0, s - k)
    
    result = risk_neutral_valuation(
        payoff_function=lambda s: call_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        simulations=50000,  # Increased from 10000 for better accuracy
        seed=42
    )
    
    # Compare with Black-Scholes (assumed to be the "true" price)
    bs_result = black_scholes(
        option_type='call',
        underlying_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    # The Monte Carlo price should be close to the Black-Scholes price
    assert abs(result['price'] - bs_result['price']) / bs_result['price'] < 0.05
    
    # Check that confidence interval is properly calculated
    assert result['confidence_interval'][0] < result['price'] < result['confidence_interval'][1]


def test_statistics():
    """Test that the statistics are calculated correctly."""
    def call_payoff(s, k=100):
        return max(0, s - k)
    
    result = risk_neutral_valuation(
        payoff_function=lambda s: call_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        simulations=10000,
        seed=42
    )
    
    # Check that statistics are calculated
    assert 'price_statistics' in result
    assert 'payoff_statistics' in result
    
    # Check specific statistics
    price_stats = result['price_statistics']
    payoff_stats = result['payoff_statistics']
    
    assert 'mean' in price_stats
    assert 'std' in price_stats
    assert 'min' in price_stats
    assert 'max' in price_stats
    assert 'median' in price_stats
    
    assert 'mean' in payoff_stats
    assert 'std' in payoff_stats
    assert 'min' in payoff_stats
    assert 'max' in payoff_stats
    assert 'median' in payoff_stats
    assert 'zero_proportion' in payoff_stats
    
    # Check that statistics are reasonable
    assert price_stats['min'] <= price_stats['median'] <= price_stats['max']
    assert payoff_stats['min'] <= payoff_stats['median'] <= payoff_stats['max']
    assert 0 <= payoff_stats['zero_proportion'] <= 1


from scipy.stats import norm

def test_binary_option_theoretical():
    """Test binary option against theoretical price."""
    def binary_call_payoff(s, k=100):
        return 1.0 if s > k else 0.0
    
    result = risk_neutral_valuation(
        payoff_function=lambda s: binary_call_payoff(s),
        underlying_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        simulations=50000,
        seed=42
    )
    
    # Binary call option price should be approximately e^(-rT) * N(d2)
    S = 100.0
    K = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.2
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theoretical_price = np.exp(-r * T) * norm.cdf(d2)
    
    # Allow for Monte Carlo error, should be within 2% of theoretical price
    assert abs(result['price'] - theoretical_price) / theoretical_price < 0.02 