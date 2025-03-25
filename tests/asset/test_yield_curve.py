"""
Tests for the yield curve construction and interpolation functions.
"""

import pytest
import numpy as np
from pypulate.asset.yield_curve import construct_yield_curve, interpolate_rate


def test_construct_yield_curve_basic():
    """Test basic yield curve construction."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    # Test with default parameters
    result = construct_yield_curve(maturities, rates)
    
    # Check that the result contains the expected keys
    assert "maturities" in result
    assert "rates" in result
    assert "interpolation_method" in result
    assert "extrapolate" in result
    assert "interpolate_func" in result
    assert "min_maturity" in result
    assert "max_maturity" in result
    assert "steepness" in result
    assert "average_rate" in result
    assert "forward_rates" in result
    
    # Check that the interpolation method is cubic by default
    assert result["interpolation_method"] == "cubic"
    
    # Check that extrapolation is disabled by default
    assert result["extrapolate"] is False
    
    # Check that the interpolation function works
    assert callable(result["interpolate_func"])
    interpolated_rate = result["interpolate_func"](4.0)
    # The interpolated rate might be a numpy scalar, so we check its value instead of type
    assert 0 <= float(interpolated_rate) <= 0.05  # Reasonable range for the test data
    
    # Check min and max maturities
    assert result["min_maturity"] == min(maturities)
    assert result["max_maturity"] == max(maturities)
    
    # Check steepness calculation
    assert result["steepness"] == rates[-1] - rates[0]
    
    # Check average rate
    assert result["average_rate"] == pytest.approx(np.mean(rates))
    
    # Check forward rates
    assert len(result["forward_rates"]) == len(maturities) - 1
    for i, forward_rate_info in enumerate(result["forward_rates"]):
        assert "start_maturity" in forward_rate_info
        assert "end_maturity" in forward_rate_info
        assert "forward_rate" in forward_rate_info
        assert forward_rate_info["start_maturity"] == maturities[i]
        assert forward_rate_info["end_maturity"] == maturities[i+1]


def test_construct_yield_curve_interpolation_methods():
    """Test different interpolation methods for yield curve construction."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    # Test linear interpolation
    linear_result = construct_yield_curve(maturities, rates, interpolation_method="linear")
    assert linear_result["interpolation_method"] == "linear"
    
    # Test cubic interpolation
    cubic_result = construct_yield_curve(maturities, rates, interpolation_method="cubic")
    assert cubic_result["interpolation_method"] == "cubic"
    
    # Test monotonic interpolation
    monotonic_result = construct_yield_curve(maturities, rates, interpolation_method="monotonic")
    assert monotonic_result["interpolation_method"] == "monotonic"
    
    # Test interpolation at a point between data points
    test_maturity = 4.0  # Between 3 and 5 years
    linear_rate = float(linear_result["interpolate_func"](test_maturity))
    cubic_rate = float(cubic_result["interpolate_func"](test_maturity))
    monotonic_rate = float(monotonic_result["interpolate_func"](test_maturity))
    
    # All interpolated rates should be reasonable
    assert 0 <= linear_rate <= 0.05
    assert 0 <= cubic_rate <= 0.05
    assert 0 <= monotonic_rate <= 0.05
    
    # For this upward sloping curve, the interpolated rate should be between the rates at 3 and 5 years
    assert rates[4] <= linear_rate <= rates[5]  # Linear should be strictly between
    # Cubic and monotonic might overshoot slightly, so we allow a small margin
    assert rates[4] - 0.001 <= cubic_rate <= rates[5] + 0.001
    assert rates[4] - 0.001 <= monotonic_rate <= rates[5] + 0.001


def test_construct_yield_curve_extrapolation():
    """Test extrapolation capabilities of yield curve construction."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032]
    
    # Test with extrapolation enabled
    result_with_extrapolation = construct_yield_curve(maturities, rates, extrapolate=True)
    assert result_with_extrapolation["extrapolate"] is True
    
    # Test with extrapolation disabled
    result_without_extrapolation = construct_yield_curve(maturities, rates, extrapolate=False)
    assert result_without_extrapolation["extrapolate"] is False
    
    # Test extrapolation to longer maturities
    extrapolated_rate_15y = float(result_with_extrapolation["interpolate_func"](15))
    
    # Extrapolated rates should be reasonable - cubic splines can extrapolate poorly for long horizons
    # so we'll test with a more reasonable maturity
    assert 0 <= extrapolated_rate_15y <= 0.05
    
    # Test that extrapolation fails when disabled
    # We need to use the interpolate_rate function which checks bounds
    with pytest.raises(ValueError):
        interpolate_rate(result_without_extrapolation, 15)


def test_construct_yield_curve_input_validation():
    """Test input validation for yield curve construction."""
    # Test mismatched lengths
    with pytest.raises(ValueError):
        construct_yield_curve([1, 2, 3], [0.01, 0.02])
    
    # Test insufficient data points
    with pytest.raises(ValueError):
        construct_yield_curve([1], [0.01])
    
    # Test non-positive maturities
    with pytest.raises(ValueError):
        construct_yield_curve([0, 1, 2], [0.01, 0.02, 0.03])
    
    # Test negative rates
    with pytest.raises(ValueError):
        construct_yield_curve([1, 2, 3], [-0.01, 0.02, 0.03])
    
    # Test unsorted maturities
    with pytest.raises(ValueError):
        construct_yield_curve([1, 3, 2], [0.01, 0.02, 0.03])
    
    # Test invalid interpolation method
    with pytest.raises(ValueError):
        construct_yield_curve([1, 2, 3], [0.01, 0.02, 0.03], interpolation_method="invalid")


def test_interpolate_rate_basic():
    """Test basic rate interpolation."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    yield_curve = construct_yield_curve(maturities, rates)
    
    # Test interpolation at a point between data points
    interpolated_rate = interpolate_rate(yield_curve, 4.0)
    assert isinstance(interpolated_rate, float)
    assert 0 <= interpolated_rate <= 0.05
    
    # Test interpolation at exact data points
    for i, maturity in enumerate(maturities):
        rate = interpolate_rate(yield_curve, maturity)
        assert rate == pytest.approx(rates[i])


def test_interpolate_rate_input_validation():
    """Test input validation for rate interpolation."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    yield_curve = construct_yield_curve(maturities, rates)
    
    # Test non-positive maturity
    with pytest.raises(ValueError):
        interpolate_rate(yield_curve, 0)
    
    # Test maturity outside range with extrapolation disabled
    with pytest.raises(ValueError):
        interpolate_rate(yield_curve, 40)
    
    # Test invalid yield curve object
    with pytest.raises(ValueError):
        interpolate_rate({}, 4.0)
    
    # Create a minimal valid yield curve object for testing
    minimal_yield_curve = {
        "interpolate_func": yield_curve["interpolate_func"],
        "extrapolate": False,
        "min_maturity": 1,
        "max_maturity": 10
    }
    
    # This should work
    interpolate_rate(minimal_yield_curve, 5.0)
    
    # But this should fail due to extrapolation
    with pytest.raises(ValueError):
        interpolate_rate(minimal_yield_curve, 15.0)


def test_yield_curve_shapes():
    """Test yield curve construction with different curve shapes."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    
    # Normal (upward sloping) yield curve
    normal_rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    # Inverted (downward sloping) yield curve
    inverted_rates = [0.035, 0.033, 0.03, 0.028, 0.026, 0.024, 0.022, 0.02, 0.018, 0.017]
    
    # Humped yield curve
    humped_rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.029, 0.028, 0.027, 0.026]
    
    # Flat yield curve
    flat_rates = [0.03] * len(maturities)
    
    # Construct yield curves for each shape
    normal_curve = construct_yield_curve(maturities, normal_rates)
    inverted_curve = construct_yield_curve(maturities, inverted_rates)
    humped_curve = construct_yield_curve(maturities, humped_rates)
    flat_curve = construct_yield_curve(maturities, flat_rates)
    
    # Check steepness calculations
    assert normal_curve["steepness"] > 0  # Upward sloping
    assert inverted_curve["steepness"] < 0  # Downward sloping
    assert abs(humped_curve["steepness"]) < abs(normal_curve["steepness"])  # Less steep
    assert flat_curve["steepness"] == 0  # Flat
    
    # Check interpolation at middle point
    middle_maturity = 4.0
    normal_rate = interpolate_rate(normal_curve, middle_maturity)
    inverted_rate = interpolate_rate(inverted_curve, middle_maturity)
    humped_rate = interpolate_rate(humped_curve, middle_maturity)
    flat_rate = interpolate_rate(flat_curve, middle_maturity)
    
    # Rates should be reasonable
    assert 0 <= normal_rate <= 0.05
    assert 0 <= inverted_rate <= 0.05
    assert 0 <= humped_rate <= 0.05
    assert flat_rate == pytest.approx(0.03)


def test_forward_rates_calculation():
    """Test forward rates calculation in yield curve construction."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    yield_curve = construct_yield_curve(maturities, rates)
    forward_rates = yield_curve["forward_rates"]
    
    # Check that we have the correct number of forward rates
    assert len(forward_rates) == len(maturities) - 1
    
    # Check a specific forward rate calculation
    # For example, the forward rate between 1 and 2 years
    # f(1,2) = ((1 + r2)^2 / (1 + r1)^1)^(1/(2-1)) - 1
    # where r1 is the 1-year rate and r2 is the 2-year rate
    r1 = rates[2]  # 1-year rate
    r2 = rates[3]  # 2-year rate
    expected_forward_rate = ((1 + r2) ** 2 / (1 + r1) ** 1) ** (1 / (2 - 1)) - 1
    
    # Find the corresponding forward rate in the result
    forward_rate_1_2 = next(fr for fr in forward_rates if fr["start_maturity"] == 1 and fr["end_maturity"] == 2)
    
    # Check that the calculated forward rate matches the expected value
    assert forward_rate_1_2["forward_rate"] == pytest.approx(expected_forward_rate)
    
    # Check that all forward rates are reasonable
    for fr in forward_rates:
        assert 0 <= fr["forward_rate"] <= 0.05 