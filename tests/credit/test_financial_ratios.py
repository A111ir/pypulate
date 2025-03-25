"""
Tests for the financial_ratios function.
"""

import pytest
from pypulate.credit.financial_ratios import financial_ratios


def test_basic_calculation():
    """Test basic financial ratios calculation with typical values."""
    result = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    
    # Test basic structure
    assert "liquidity" in result
    assert "solvency" in result
    assert "profitability" in result
    assert "coverage" in result
    assert "efficiency" in result
    assert "overall_assessment" in result
    
    # Test specific ratios
    assert result["liquidity"]["current_ratio"] == 2.0
    assert result["solvency"]["debt_ratio"] == 0.4
    assert result["solvency"]["debt_to_equity"] == 2/3
    assert result["profitability"]["return_on_assets"] == 0.1
    assert result["profitability"]["return_on_equity"] == pytest.approx(1/6)
    assert result["coverage"]["interest_coverage"] == 5.0
    assert result["efficiency"]["asset_turnover"] == 0.8


def test_liquidity_assessments():
    """Test liquidity assessment categories."""
    # Strong: current_ratio >= 2
    result_strong = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result_strong["liquidity"]["assessment"] == "Strong"
    
    # Adequate: 1 <= current_ratio < 2
    result_adequate = financial_ratios(
        current_assets=150000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result_adequate["liquidity"]["assessment"] == "Adequate"
    
    # Weak: current_ratio < 1
    result_weak = financial_ratios(
        current_assets=90000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result_weak["liquidity"]["assessment"] == "Weak"


def test_solvency_assessments():
    """Test solvency assessment categories."""
    # Strong: debt_ratio <= 0.4
    result_strong = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result_strong["solvency"]["assessment"] == "Strong"
    
    # Adequate: 0.4 < debt_ratio <= 0.6
    result_adequate = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=500000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=500000,
        sales=800000
    )
    assert result_adequate["solvency"]["assessment"] == "Adequate"
    
    # Weak: debt_ratio > 0.6
    result_weak = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=700000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=300000,
        sales=800000
    )
    assert result_weak["solvency"]["assessment"] == "Weak"


def test_profitability_assessments():
    """Test profitability assessment categories."""
    # Strong: return_on_equity >= 0.15
    result_strong = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=90000,
        total_equity=600000,
        sales=800000
    )
    assert result_strong["profitability"]["return_on_equity"] == 0.15
    assert result_strong["profitability"]["assessment"] == "Strong"
    
    # Adequate: 0.08 <= return_on_equity < 0.15
    result_adequate = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=60000,
        total_equity=600000,
        sales=800000
    )
    assert result_adequate["profitability"]["return_on_equity"] == 0.1
    assert result_adequate["profitability"]["assessment"] == "Adequate"
    
    # Weak: return_on_equity < 0.08
    result_weak = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=40000,
        total_equity=600000,
        sales=800000
    )
    assert result_weak["profitability"]["return_on_equity"] == pytest.approx(1/15)
    assert result_weak["profitability"]["assessment"] == "Weak"


def test_coverage_assessments():
    """Test coverage assessment categories."""
    # Strong: interest_coverage >= 3
    result_strong = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=90000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result_strong["coverage"]["interest_coverage"] == 3.0
    assert result_strong["coverage"]["assessment"] == "Strong"
    
    # Adequate: 1.5 <= interest_coverage < 3
    result_adequate = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=60000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result_adequate["coverage"]["interest_coverage"] == 2.0
    assert result_adequate["coverage"]["assessment"] == "Adequate"
    
    # Weak: interest_coverage < 1.5
    result_weak = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=30000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result_weak["coverage"]["interest_coverage"] == 1.0
    assert result_weak["coverage"]["assessment"] == "Weak"


def test_overall_assessment():
    """Test overall assessment classification."""
    # Strong: 3+ categories are "Strong"
    result_strong = financial_ratios(
        current_assets=200000,  # Strong
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,  # Strong
        ebit=150000,  # Strong
        interest_expense=30000,
        net_income=90000,  # Strong
        total_equity=600000,
        sales=800000
    )
    assert result_strong["overall_assessment"] == "Strong financial position"
    
    # Weak: 3+ categories are "Weak"
    result_weak = financial_ratios(
        current_assets=50000,  # Weak
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=700000,  # Weak
        ebit=20000,  # Weak
        interest_expense=30000,
        net_income=20000,  # Weak
        total_equity=300000,
        sales=800000
    )
    assert result_weak["overall_assessment"] == "Weak financial position"
    
    # Adequate: Neither 3+ Strong nor 3+ Weak
    result_adequate = financial_ratios(
        current_assets=50000,  # Weak
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=450000,  # Adequate
        ebit=90000,  # Strong
        interest_expense=30000,
        net_income=60000,  # Adequate
        total_equity=550000,
        sales=800000
    )
    assert result_adequate["overall_assessment"] == "Adequate financial position"


def test_efficiency_ratio():
    """Test the efficiency ratio calculation."""
    result = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    assert result["efficiency"]["asset_turnover"] == 0.8
    
    result_high = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=2000000
    )
    assert result_high["efficiency"]["asset_turnover"] == 2.0


def test_edge_cases():
    """Test edge cases for financial ratios calculation."""
    # Ratios exactly at threshold boundaries
    boundary_result = financial_ratios(
        current_assets=200000,  # current_ratio = 2.0 (Strong boundary)
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,  # debt_ratio = 0.4 (Strong boundary)
        ebit=45000,  # interest_coverage = 1.5 (Adequate boundary)
        interest_expense=30000,
        net_income=48000,  # return_on_equity = 0.08 (Adequate boundary)
        total_equity=600000,
        sales=800000
    )
    
    assert boundary_result["liquidity"]["assessment"] == "Strong"
    assert boundary_result["solvency"]["assessment"] == "Strong"
    assert boundary_result["profitability"]["assessment"] == "Adequate"
    assert boundary_result["coverage"]["assessment"] == "Adequate"
    
    # Very high ratios
    high_result = financial_ratios(
        current_assets=1000000,
        current_liabilities=100000,
        total_assets=2000000,
        total_liabilities=200000,
        ebit=500000,
        interest_expense=10000,
        net_income=400000,
        total_equity=1800000,
        sales=3000000
    )
    
    assert high_result["liquidity"]["current_ratio"] == 10.0
    assert high_result["coverage"]["interest_coverage"] == 50.0
    assert high_result["overall_assessment"] == "Strong financial position"


def test_zero_values():
    """Test handling of zero values that could cause division by zero."""
    # Zero current liabilities
    with pytest.raises(ZeroDivisionError):
        financial_ratios(
            current_assets=200000,
            current_liabilities=0,  # Will cause ZeroDivisionError in current_ratio
            total_assets=1000000,
            total_liabilities=400000,
            ebit=150000,
            interest_expense=30000,
            net_income=100000,
            total_equity=600000,
            sales=800000
        )
    
    # Zero interest expense
    with pytest.raises(ZeroDivisionError):
        financial_ratios(
            current_assets=200000,
            current_liabilities=100000,
            total_assets=1000000,
            total_liabilities=400000,
            ebit=150000,
            interest_expense=0,  # Will cause ZeroDivisionError in interest_coverage
            net_income=100000,
            total_equity=600000,
            sales=800000
        )
    
    # Zero equity
    with pytest.raises(ZeroDivisionError):
        financial_ratios(
            current_assets=200000,
            current_liabilities=100000,
            total_assets=1000000,
            total_liabilities=400000,
            ebit=150000,
            interest_expense=30000,
            net_income=100000,
            total_equity=0,  # Will cause ZeroDivisionError in return_on_equity and debt_to_equity
            sales=800000
        )


def test_result_structure():
    """Test that the result structure contains all expected fields and assessments."""
    result = financial_ratios(
        current_assets=200000,
        current_liabilities=100000,
        total_assets=1000000,
        total_liabilities=400000,
        ebit=150000,
        interest_expense=30000,
        net_income=100000,
        total_equity=600000,
        sales=800000
    )
    
    # Check liquidity section
    assert "current_ratio" in result["liquidity"]
    assert "assessment" in result["liquidity"]
    
    # Check solvency section
    assert "debt_ratio" in result["solvency"]
    assert "debt_to_equity" in result["solvency"]
    assert "assessment" in result["solvency"]
    
    # Check profitability section
    assert "return_on_assets" in result["profitability"]
    assert "return_on_equity" in result["profitability"]
    assert "assessment" in result["profitability"]
    
    # Check coverage section
    assert "interest_coverage" in result["coverage"]
    assert "assessment" in result["coverage"]
    
    # Check efficiency section
    assert "asset_turnover" in result["efficiency"]
    
    # Check overall assessment
    assert result["overall_assessment"] in [
        "Strong financial position",
        "Adequate financial position",
        "Weak financial position"
    ] 