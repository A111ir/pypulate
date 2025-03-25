import pytest
import math
import numpy as np
from pypulate.pricing.time_based_pricing import calculate_time_based_price


def test_basic_hourly_pricing():
    # Test basic hourly pricing
    result = calculate_time_based_price(100, 2.5, 'hour')
    expected = 250.0  # 2.5 hours at $100/hour
    assert result == expected


def test_minute_pricing():
    # Test pricing per minute
    result = calculate_time_based_price(2, 30, 'minute')
    expected = 60.0  # 30 minutes at $2/minute
    assert result == expected


def test_day_pricing():
    # Test pricing per day
    result = calculate_time_based_price(500, 3, 'day')
    expected = 1500.0  # 3 days at $500/day
    assert result == expected


def test_minimum_duration():
    # Test minimum billable duration
    result = calculate_time_based_price(100, 0.5, 'hour', minimum_duration=1.0)
    expected = 100.0  # 0.5 hours adjusted to 1 hour minimum at $100/hour
    assert result == expected


def test_rounding_up():
    # Test rounding up
    result = calculate_time_based_price(10, 2.3, 'hour', rounding_method='up')
    expected = 23.0  # 2.3 hours at $10/hour = $23, rounded up to $23
    assert result == expected


def test_rounding_down():
    # Test rounding down
    result = calculate_time_based_price(10, 2.7, 'hour', rounding_method='down')
    expected = 27.0  # 2.7 hours at $10/hour = $27, rounded down to $27
    assert result == expected


def test_rounding_nearest():
    # Test rounding to nearest
    result1 = calculate_time_based_price(10, 2.3, 'hour', rounding_method='nearest')
    expected1 = 23.0  # 2.3 hours at $10/hour = $23, rounded to nearest is $23
    assert result1 == expected1
    
    result2 = calculate_time_based_price(10, 2.7, 'hour', rounding_method='nearest')
    expected2 = 27.0  # 2.7 hours at $10/hour = $27, rounded to nearest is $27
    assert result2 == expected2


def test_zero_duration():
    # Test with zero duration (should use minimum duration)
    result = calculate_time_based_price(100, 0, 'hour', minimum_duration=0.5)
    expected = 50.0  # 0 hours adjusted to 0.5 hour minimum at $100/hour
    assert result == expected


def test_zero_base_price():
    # Test with zero base price
    result = calculate_time_based_price(0, 5, 'hour')
    expected = 0.0  # 5 hours at $0/hour
    assert result == expected


def test_invalid_time_unit():
    # Test with invalid time unit
    with pytest.raises(ValueError) as excinfo:
        calculate_time_based_price(100, 1, 'week')
    assert "Unsupported time unit" in str(excinfo.value)


def test_invalid_rounding_method():
    # Test with invalid rounding method
    with pytest.raises(ValueError) as excinfo:
        calculate_time_based_price(100, 1, 'hour', rounding_method='invalid')
    assert "Unsupported rounding method" in str(excinfo.value)


def test_minimum_duration_with_rounding():
    # Test combination of minimum duration and rounding
    result = calculate_time_based_price(100, 0.3, 'hour', minimum_duration=0.5, rounding_method='up')
    expected = 50.0  # 0.3 hours adjusted to 0.5 hour minimum at $100/hour, then rounded up
    assert result == expected 