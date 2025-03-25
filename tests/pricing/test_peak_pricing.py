import pytest
from pypulate.pricing.peak_pricing import calculate_peak_pricing


def test_peak_hours_monday():
    # Test during peak hours on Monday
    base_price = 100.0
    usage_time = "10:00"
    peak_hours = {"monday": ("09:00", "17:00")}
    result = calculate_peak_pricing(base_price, usage_time, peak_hours)
    assert result == 150.0  # 100 * 1.5 (peak multiplier)


def test_off_peak_hours_monday():
    # Test during off-peak hours on Monday
    base_price = 100.0
    usage_time = "20:00"
    peak_hours = {"monday": ("09:00", "17:00")}
    result = calculate_peak_pricing(base_price, usage_time, peak_hours)
    assert result == 80.0  # 100 * 0.8 (off-peak multiplier)


def test_peak_hours_different_days():
    # Test that peak hours work for different days of the week
    base_price = 100.0
    
    # Test Tuesday
    peak_hours = {"tuesday": ("09:00", "17:00")}
    result = calculate_peak_pricing(base_price, "12:00", peak_hours)
    assert result == 150.0
    
    # Test Friday
    peak_hours = {"friday": ("09:00", "17:00")}
    result = calculate_peak_pricing(base_price, "16:59", peak_hours)
    assert result == 150.0
    
    # Test Sunday
    peak_hours = {"sunday": ("09:00", "17:00")}
    result = calculate_peak_pricing(base_price, "14:30", peak_hours)
    assert result == 150.0


def test_multiple_peak_periods():
    # Test with multiple peak periods defined
    base_price = 100.0
    usage_time = "10:00"
    peak_hours = {
        "monday": ("09:00", "17:00"),
        "tuesday": ("08:00", "12:00"),
        "wednesday": ("13:00", "18:00")
    }
    # Should apply peak pricing because time is within Monday's peak hours
    result = calculate_peak_pricing(base_price, usage_time, peak_hours)
    assert result == 150.0


def test_boundary_conditions():
    # Test at the boundaries of peak hours
    base_price = 100.0
    peak_hours = {"monday": ("09:00", "17:00")}
    
    # At the start of peak hours
    result = calculate_peak_pricing(base_price, "09:00", peak_hours)
    assert result == 150.0
    
    # Just before the start of peak hours
    result = calculate_peak_pricing(base_price, "08:59", peak_hours)
    assert result == 80.0
    
    # Just before the end of peak hours
    result = calculate_peak_pricing(base_price, "16:59", peak_hours)
    assert result == 150.0
    
    # At the end of peak hours (exclusive)
    result = calculate_peak_pricing(base_price, "17:00", peak_hours)
    assert result == 80.0


def test_custom_multipliers():
    # Test with custom peak and off-peak multipliers
    base_price = 100.0
    usage_time = "10:00"
    peak_hours = {"monday": ("09:00", "17:00")}
    
    # Custom peak multiplier
    result = calculate_peak_pricing(base_price, usage_time, peak_hours, peak_multiplier=2.0)
    assert result == 200.0  # 100 * 2.0
    
    # Custom off-peak multiplier
    usage_time = "20:00"  # Off-peak
    result = calculate_peak_pricing(base_price, usage_time, peak_hours, off_peak_multiplier=0.5)
    assert result == 50.0  # 100 * 0.5


def test_zero_base_price():
    # Test with zero base price
    base_price = 0.0
    usage_time = "10:00"
    peak_hours = {"monday": ("09:00", "17:00")}
    result = calculate_peak_pricing(base_price, usage_time, peak_hours)
    assert result == 0.0


def test_no_peak_hours():
    # Test when no peak hours are defined
    base_price = 100.0
    usage_time = "10:00"
    peak_hours = {}
    # Should use off-peak pricing as no peak hours are matched
    result = calculate_peak_pricing(base_price, usage_time, peak_hours)
    assert result == 80.0  # 100 * 0.8


def test_minute_precision():
    # Test that minutes are considered in peak hour calculations
    base_price = 100.0
    peak_hours = {"monday": ("09:30", "10:30")}
    
    # Before peak hours by minutes
    result = calculate_peak_pricing(base_price, "09:29", peak_hours)
    assert result == 80.0
    
    # Within peak hours including minutes
    result = calculate_peak_pricing(base_price, "09:45", peak_hours)
    assert result == 150.0
    
    # After peak hours by minutes
    result = calculate_peak_pricing(base_price, "10:30", peak_hours)
    assert result == 80.0 