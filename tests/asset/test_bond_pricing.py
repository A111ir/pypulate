"""
Tests for bond pricing and fixed income analysis functions.
"""

import pytest
import numpy as np
from pypulate.asset import price_bond, yield_to_maturity, duration_convexity


def test_basic_bond_pricing():
    """Test basic bond pricing."""
    result = price_bond(
        face_value=1000.0,
        coupon_rate=0.05,
        years_to_maturity=10.0,
        yield_to_maturity=0.05,
        frequency=2
    )
    
    # When YTM equals coupon rate, bond should trade at par
    assert abs(result['price'] - result['face_value']) < 0.01
    assert result['status'] == "At par"
    assert abs(result['current_yield'] - result['coupon_rate']) < 0.0001


def test_premium_bond():
    """Test pricing of premium bonds (YTM < coupon rate)."""
    result = price_bond(
        face_value=1000.0,
        coupon_rate=0.06,
        years_to_maturity=10.0,
        yield_to_maturity=0.04,
        frequency=2
    )
    
    assert result['price'] > result['face_value']
    assert result['status'] == "Trading at premium"
    assert result['current_yield'] < result['coupon_rate']


def test_discount_bond():
    """Test pricing of discount bonds (YTM > coupon rate)."""
    result = price_bond(
        face_value=1000.0,
        coupon_rate=0.04,
        years_to_maturity=10.0,
        yield_to_maturity=0.06,
        frequency=2
    )
    
    assert result['price'] < result['face_value']
    assert result['status'] == "Trading at discount"
    assert result['current_yield'] > result['coupon_rate']


def test_zero_coupon_bond():
    """Test pricing of zero-coupon bonds."""
    result = price_bond(
        face_value=1000.0,
        coupon_rate=0.0,
        years_to_maturity=10.0,
        yield_to_maturity=0.05,
        frequency=2
    )
    
    # Manual calculation of zero-coupon bond price
    expected_price = 1000.0 / (1 + 0.05/2) ** (10 * 2)
    assert abs(result['price'] - expected_price) < 0.01
    assert result['coupon_payment'] == 0.0


def test_bond_pricing_input_validation():
    """Test input validation for bond pricing."""
    with pytest.raises(ValueError):
        price_bond(
            face_value=-1000.0,  # Invalid negative face value
            coupon_rate=0.05,
            years_to_maturity=10.0,
            yield_to_maturity=0.05
        )
    
    with pytest.raises(ValueError):
        price_bond(
            face_value=1000.0,
            coupon_rate=-0.05,  # Invalid negative coupon rate
            years_to_maturity=10.0,
            yield_to_maturity=0.05
        )
    
    with pytest.raises(ValueError):
        price_bond(
            face_value=1000.0,
            coupon_rate=0.05,
            years_to_maturity=-10.0,  # Invalid negative maturity
            yield_to_maturity=0.05
        )
    
    with pytest.raises(ValueError):
        price_bond(
            face_value=1000.0,
            coupon_rate=0.05,
            years_to_maturity=10.0,
            yield_to_maturity=0.05,
            frequency=0  # Invalid frequency
        )


def test_basic_ytm():
    """Test basic yield to maturity calculation."""
    # First calculate a bond price
    bond_params = {
        'face_value': 1000.0,
        'coupon_rate': 0.05,
        'years_to_maturity': 10.0,
        'yield_to_maturity': 0.06,
        'frequency': 2
    }
    price_result = price_bond(**bond_params)
    
    # Then recover the YTM from the price
    ytm_result = yield_to_maturity(
        price=price_result['price'],
        face_value=1000.0,
        coupon_rate=0.05,
        years_to_maturity=10.0,
        frequency=2
    )
    
    # The recovered YTM should match the original
    assert abs(ytm_result['yield_to_maturity'] - bond_params['yield_to_maturity']) < 1e-4


def test_ytm_input_validation():
    """Test input validation for yield to maturity calculation."""
    with pytest.raises(ValueError):
        yield_to_maturity(
            price=-1000.0,  # Invalid negative price
            face_value=1000.0,
            coupon_rate=0.05,
            years_to_maturity=10.0
        )
    
    with pytest.raises(ValueError):
        yield_to_maturity(
            price=1000.0,
            face_value=-1000.0,  # Invalid negative face value
            coupon_rate=0.05,
            years_to_maturity=10.0
        )


def test_ytm_convergence():
    """Test YTM calculation with extreme prices."""
    params = {
        'face_value': 1000.0,
        'coupon_rate': 0.05,
        'years_to_maturity': 10.0,
        'frequency': 2
    }
    
    # Test with high price (low YTM)
    high_price_result = yield_to_maturity(
        price=1200.0,
        **params
    )
    assert 'error' not in high_price_result
    assert high_price_result['yield_to_maturity'] > 0
    
    # Test with low price (high YTM)
    low_price_result = yield_to_maturity(
        price=800.0,
        **params
    )
    assert 'error' not in low_price_result
    assert low_price_result['yield_to_maturity'] > 0


def test_basic_duration_convexity():
    """Test basic duration and convexity calculation."""
    result = duration_convexity(
        face_value=1000.0,
        coupon_rate=0.05,
        years_to_maturity=10.0,
        yield_to_maturity=0.05,
        frequency=2
    )
    
    # Basic sanity checks
    assert result['macaulay_duration'] > 0
    assert result['modified_duration'] > 0
    assert result['convexity'] > 0
    assert result['macaulay_duration'] > result['modified_duration']  # Always true when yield > 0


def test_duration_relationships():
    """Test relationships between duration and bond characteristics."""
    base_params = {
        'face_value': 1000.0,
        'years_to_maturity': 10.0,
        'yield_to_maturity': 0.05,
        'frequency': 2
    }
    
    # Higher coupon rate should lead to lower duration
    high_coupon = duration_convexity(**base_params, coupon_rate=0.08)
    low_coupon = duration_convexity(**base_params, coupon_rate=0.03)
    assert high_coupon['macaulay_duration'] < low_coupon['macaulay_duration']
    
    # Zero-coupon bond should have duration equal to maturity
    zero_coupon = duration_convexity(**base_params, coupon_rate=0.0)
    assert abs(zero_coupon['macaulay_duration'] - base_params['years_to_maturity']) < 0.01


def test_duration_convexity_input_validation():
    """Test input validation for duration and convexity calculation."""
    with pytest.raises(ValueError):
        duration_convexity(
            face_value=-1000.0,  # Invalid negative face value
            coupon_rate=0.05,
            years_to_maturity=10.0,
            yield_to_maturity=0.05
        )
    
    with pytest.raises(ValueError):
        duration_convexity(
            face_value=1000.0,
            coupon_rate=0.05,
            years_to_maturity=10.0,
            yield_to_maturity=-0.05  # Invalid negative yield
        )


def test_price_sensitivity():
    """Test price sensitivity calculations."""
    result = duration_convexity(
        face_value=1000.0,
        coupon_rate=0.05,
        years_to_maturity=10.0,
        yield_to_maturity=0.05,
        frequency=2
    )
    
    # Price change for small yield changes should be approximately linear
    price_change_1bp = result['price_change_1bp']
    assert price_change_1bp < 0  # Price should decrease when yield increases
    
    # Convexity adjustment should make actual price change less negative
    price_change_100bp = result['price_change_100bp']
    price_change_with_convexity = result['price_change_100bp_with_convexity']
    assert price_change_with_convexity > price_change_100bp


def test_numerical_stability():
    """Test numerical stability with extreme parameters."""
    result = duration_convexity(
        face_value=1000000.0,  # Very high face value
        coupon_rate=0.10,      # High coupon rate
        years_to_maturity=30.0, # Long maturity
        yield_to_maturity=0.15, # High yield
        frequency=12           # Monthly payments
    )
    
    assert np.isfinite(result['macaulay_duration'])
    assert np.isfinite(result['modified_duration'])
    assert np.isfinite(result['convexity'])
    assert np.isfinite(result['price'])
    assert np.isfinite(result['price_change_100bp_with_convexity']) 