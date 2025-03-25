"""
Tests for the debt_service_coverage_ratio function.
"""

import pytest
from pypulate.credit.debt_service_coverage_ratio import debt_service_coverage_ratio


def test_excellent_rating():
    """Test with values that should result in 'Excellent' rating (DSCR >= 1.5)."""
    result = debt_service_coverage_ratio(150000, 100000)
    
    assert result["dscr"] == 1.5
    assert result["assessment"] == "Strong coverage, low risk"
    assert result["rating"] == "Excellent"


def test_good_rating():
    """Test with values that should result in 'Good' rating (1.25 <= DSCR < 1.5)."""
    result = debt_service_coverage_ratio(130000, 100000)
    
    assert result["dscr"] == 1.3
    assert result["assessment"] == "Sufficient coverage, acceptable risk"
    assert result["rating"] == "Good"


def test_fair_rating():
    """Test with values that should result in 'Fair' rating (1.0 <= DSCR < 1.25)."""
    result = debt_service_coverage_ratio(120000, 100000)
    
    assert result["dscr"] == 1.2
    assert result["assessment"] == "Barely sufficient, moderate risk"
    assert result["rating"] == "Fair"


def test_poor_rating():
    """Test with values that should result in 'Poor' rating (DSCR < 1.0)."""
    result = debt_service_coverage_ratio(90000, 100000)
    
    assert result["dscr"] == 0.9
    assert result["assessment"] == "Negative cash flow, high risk"
    assert result["rating"] == "Poor"


def test_exact_thresholds():
    """Test with values exactly at the threshold boundaries."""
    # At 1.5 threshold (Excellent/Good boundary)
    result_1 = debt_service_coverage_ratio(150000, 100000)
    assert result_1["rating"] == "Excellent"
    
    # At 1.25 threshold (Good/Fair boundary)
    result_2 = debt_service_coverage_ratio(125000, 100000)
    assert result_2["rating"] == "Good"
    
    # At 1.0 threshold (Fair/Poor boundary)
    result_3 = debt_service_coverage_ratio(100000, 100000)
    assert result_3["rating"] == "Fair"


def test_zero_debt_service():
    """Test with zero debt service (should raise ZeroDivisionError)."""
    with pytest.raises(ZeroDivisionError):
        debt_service_coverage_ratio(100000, 0)


def test_negative_values():
    """Test with negative values for net operating income and debt service."""
    # Negative net operating income
    result_1 = debt_service_coverage_ratio(-100000, 100000)
    assert result_1["dscr"] == -1.0
    assert result_1["rating"] == "Poor"
    
    # Negative debt service
    result_2 = debt_service_coverage_ratio(100000, -100000)
    assert result_2["dscr"] == -1.0
    assert result_2["rating"] == "Poor"
    
    # Both negative
    result_3 = debt_service_coverage_ratio(-100000, -100000)
    assert result_3["dscr"] == 1.0
    assert result_3["rating"] == "Fair"


def test_large_numbers():
    """Test with very large numbers to ensure numerical stability."""
    result = debt_service_coverage_ratio(1e9, 1e8)
    
    assert result["dscr"] == 10.0
    assert result["rating"] == "Excellent"


def test_small_numbers():
    """Test with very small numbers to ensure numerical stability."""
    result = debt_service_coverage_ratio(1e-6, 1e-6)
    
    assert result["dscr"] == 1.0
    assert result["rating"] == "Fair"


def test_equal_values():
    """Test with equal net operating income and debt service."""
    result = debt_service_coverage_ratio(100000, 100000)
    
    assert result["dscr"] == 1.0
    assert result["rating"] == "Fair" 