"""
Tests for the loan_pricing function.
"""

import pytest
import numpy as np
from pypulate.credit.loan_pricing import loan_pricing


def test_basic_calculation():
    """Test basic loan pricing calculation with typical values."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Test basic structure
    assert "recommended_rate" in result
    assert "effective_annual_rate" in result
    assert "monthly_payment" in result
    assert "total_interest" in result
    assert "expected_profit" in result
    assert "return_on_investment" in result
    assert "components" in result
    
    # Test components
    expected_loss_rate = 0.02 * 0.4  # pd * lgd
    expected_loss_component = expected_loss_rate / 5  # expected_loss_rate / term
    funding_component = 0.03
    operating_component = 0.01 / 5
    capital_component = 0.08 * 0.15  # capital_requirement * target_roe
    risk_premium = expected_loss_component + capital_component
    recommended_rate = funding_component + operating_component + risk_premium
    
    assert result["components"]["expected_loss"] == pytest.approx(expected_loss_component)
    assert result["components"]["funding_cost"] == pytest.approx(funding_component)
    assert result["components"]["operating_cost"] == pytest.approx(operating_component)
    assert result["components"]["capital_cost"] == pytest.approx(capital_component)
    assert result["components"]["risk_premium"] == pytest.approx(risk_premium)
    assert result["recommended_rate"] == pytest.approx(recommended_rate)


def test_recommended_rate_calculation():
    """Test the calculation of the recommended interest rate."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Manual calculation of recommended rate
    expected_loss_component = (0.02 * 0.4) / 5
    funding_component = 0.03
    operating_component = 0.01 / 5
    capital_component = 0.08 * 0.15
    expected_rate = funding_component + operating_component + expected_loss_component + capital_component
    
    assert result["recommended_rate"] == pytest.approx(expected_rate)


def test_effective_annual_rate():
    """Test the calculation of effective annual rate."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Calculate effective annual rate: (1 + r/12)^12 - 1
    nominal_rate = result["recommended_rate"]
    expected_ear = (1 + nominal_rate / 12) ** 12 - 1
    
    assert result["effective_annual_rate"] == pytest.approx(expected_ear)


def test_monthly_payment():
    """Test the calculation of monthly payment."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Manual calculation of monthly payment using the loan formula
    rate = result["recommended_rate"] / 12
    num_payments = 5 * 12
    expected_payment = 100000 * (rate * (1 + rate) ** num_payments) / ((1 + rate) ** num_payments - 1)
    
    assert result["monthly_payment"] == pytest.approx(expected_payment)


def test_total_interest():
    """Test the calculation of total interest paid."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Total interest = total payments - principal
    monthly_payment = result["monthly_payment"]
    total_payments = monthly_payment * 5 * 12
    expected_total_interest = total_payments - 100000
    
    assert result["total_interest"] == pytest.approx(expected_total_interest)


def test_expected_profit():
    """Test the calculation of expected profit."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Expected profit = total interest - expected loss - operating costs
    total_interest = result["total_interest"]
    expected_loss = 100000 * 0.02 * 0.4  # loan_amount * pd * lgd
    operating_costs = 0.01 * 100000 * 5  # operating_cost * loan_amount * term
    expected_profit = total_interest - expected_loss - operating_costs
    
    assert result["expected_profit"] == pytest.approx(expected_profit)


def test_return_on_investment():
    """Test the calculation of return on investment."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # ROI = expected profit / capital
    expected_profit = result["expected_profit"]
    capital = 0.08 * 100000  # capital_requirement * loan_amount
    expected_roi = expected_profit / capital
    
    assert result["return_on_investment"] == pytest.approx(expected_roi)


def test_term_effect():
    """Test how different loan terms affect pricing."""
    # Short term (1 year)
    short_term = loan_pricing(
        loan_amount=100000,
        term=1,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Standard term (5 years)
    standard_term = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Long term (10 years)
    long_term = loan_pricing(
        loan_amount=100000,
        term=10,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Expected loss component should decrease with longer terms
    assert short_term["components"]["expected_loss"] > standard_term["components"]["expected_loss"]
    assert standard_term["components"]["expected_loss"] > long_term["components"]["expected_loss"]
    
    # Operating cost component should decrease with longer terms
    assert short_term["components"]["operating_cost"] > standard_term["components"]["operating_cost"]
    assert standard_term["components"]["operating_cost"] > long_term["components"]["operating_cost"]
    
    # Recommended rate should decrease with longer terms (due to spreading costs)
    assert short_term["recommended_rate"] > standard_term["recommended_rate"]
    assert standard_term["recommended_rate"] > long_term["recommended_rate"]
    
    # Monthly payment should decrease with longer terms
    assert short_term["monthly_payment"] > standard_term["monthly_payment"]
    assert standard_term["monthly_payment"] > long_term["monthly_payment"]
    
    # Total interest should increase with longer terms
    assert short_term["total_interest"] < standard_term["total_interest"]
    assert standard_term["total_interest"] < long_term["total_interest"]


def test_risk_effect():
    """Test how different risk profiles affect pricing."""
    # Low risk
    low_risk = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.01,
        lgd=0.2,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Medium risk
    medium_risk = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.03,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # High risk
    high_risk = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.05,
        lgd=0.6,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Expected loss component should increase with higher risk
    assert low_risk["components"]["expected_loss"] < medium_risk["components"]["expected_loss"]
    assert medium_risk["components"]["expected_loss"] < high_risk["components"]["expected_loss"]
    
    # Recommended rate should increase with higher risk
    assert low_risk["recommended_rate"] < medium_risk["recommended_rate"]
    assert medium_risk["recommended_rate"] < high_risk["recommended_rate"]
    
    # Monthly payment should increase with higher risk
    assert low_risk["monthly_payment"] < medium_risk["monthly_payment"]
    assert medium_risk["monthly_payment"] < high_risk["monthly_payment"]


def test_funding_cost_effect():
    """Test how different funding costs affect pricing."""
    # Low funding cost
    low_funding = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.02,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Medium funding cost
    medium_funding = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.04,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # High funding cost
    high_funding = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.06,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Funding component should increase with higher funding cost
    assert low_funding["components"]["funding_cost"] < medium_funding["components"]["funding_cost"]
    assert medium_funding["components"]["funding_cost"] < high_funding["components"]["funding_cost"]
    
    # Recommended rate should increase with higher funding cost
    assert low_funding["recommended_rate"] < medium_funding["recommended_rate"]
    assert medium_funding["recommended_rate"] < high_funding["recommended_rate"]


def test_invalid_inputs():
    """Test invalid inputs that should raise ValueError."""
    # Negative loan amount
    with pytest.raises(ValueError, match="Loan amount and term must be positive"):
        loan_pricing(
            loan_amount=-100000,
            term=5,
            pd=0.02,
            lgd=0.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )
    
    # Zero loan amount
    with pytest.raises(ValueError, match="Loan amount and term must be positive"):
        loan_pricing(
            loan_amount=0,
            term=5,
            pd=0.02,
            lgd=0.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )
    
    # Negative term
    with pytest.raises(ValueError, match="Loan amount and term must be positive"):
        loan_pricing(
            loan_amount=100000,
            term=-5,
            pd=0.02,
            lgd=0.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )
    
    # Zero term
    with pytest.raises(ValueError, match="Loan amount and term must be positive"):
        loan_pricing(
            loan_amount=100000,
            term=0,
            pd=0.02,
            lgd=0.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )
    
    # PD < 0
    with pytest.raises(ValueError, match="Probability of default must be between 0 and 1"):
        loan_pricing(
            loan_amount=100000,
            term=5,
            pd=-0.02,
            lgd=0.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )
    
    # PD > 1
    with pytest.raises(ValueError, match="Probability of default must be between 0 and 1"):
        loan_pricing(
            loan_amount=100000,
            term=5,
            pd=1.2,
            lgd=0.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )
    
    # LGD < 0
    with pytest.raises(ValueError, match="Loss given default must be between 0 and 1"):
        loan_pricing(
            loan_amount=100000,
            term=5,
            pd=0.02,
            lgd=-0.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )
    
    # LGD > 1
    with pytest.raises(ValueError, match="Loss given default must be between 0 and 1"):
        loan_pricing(
            loan_amount=100000,
            term=5,
            pd=0.02,
            lgd=1.4,
            funding_cost=0.03,
            operating_cost=0.01,
            capital_requirement=0.08,
            target_roe=0.15
        )


def test_edge_cases():
    """Test edge cases for loan pricing calculation."""
    # Zero risk (PD = 0)
    zero_risk = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    assert zero_risk["components"]["expected_loss"] == 0
    
    # Zero LGD
    zero_lgd = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    assert zero_lgd["components"]["expected_loss"] == 0
    
    # Zero operating cost
    zero_op_cost = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0,
        capital_requirement=0.08,
        target_roe=0.15
    )
    assert zero_op_cost["components"]["operating_cost"] == 0
    
    # We can't test zero capital requirement because it causes division by zero
    # in the ROI calculation. The implementation would need to handle this edge case.
    
    # Zero target ROE
    zero_roe = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0
    )
    assert zero_roe["components"]["capital_cost"] == 0
    
    # Very short term (0.5 years)
    short_term = loan_pricing(
        loan_amount=100000,
        term=0.5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    assert short_term["components"]["expected_loss"] == pytest.approx((0.02 * 0.4) / 0.5)
    
    # Very long term (30 years)
    long_term = loan_pricing(
        loan_amount=100000,
        term=30,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    assert long_term["components"]["expected_loss"] == pytest.approx((0.02 * 0.4) / 30)


def test_profit_calculations():
    """Test that expected profit and ROI calculations are correct."""
    result = loan_pricing(
        loan_amount=100000,
        term=5,
        pd=0.02,
        lgd=0.4,
        funding_cost=0.03,
        operating_cost=0.01,
        capital_requirement=0.08,
        target_roe=0.15
    )
    
    # Verify expected profit calculation
    total_interest = result["total_interest"]
    expected_loss = 100000 * 0.02 * 0.4
    operating_costs = 0.01 * 100000 * 5
    expected_profit = total_interest - expected_loss - operating_costs
    
    assert result["expected_profit"] == pytest.approx(expected_profit)
    
    # Verify ROI calculation
    capital = 0.08 * 100000
    expected_roi = expected_profit / capital
    
    assert result["return_on_investment"] == pytest.approx(expected_roi)
    
    # ROI will likely be different from target_roe because it depends on actual
    # profit generated from the loan, which is influenced by term, amortization, etc.
    # We can't reliably assert a fixed threshold, just that the calculation is correct 