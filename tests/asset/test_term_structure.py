"""
Tests for the term structure models.
"""

import pytest
import numpy as np
from pypulate.asset.term_structure import nelson_siegel, svensson


def test_nelson_siegel_basic():
    """Test basic Nelson-Siegel model fitting."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    result = nelson_siegel(maturities, rates)
    
    # Check that the result contains the expected keys
    assert "parameters" in result
    assert "parameter_names" in result
    assert "predict_func" in result
    assert "fitted_rates" in result
    assert "residuals" in result
    assert "r_squared" in result
    assert "rmse" in result
    assert "short_rate" in result
    assert "long_rate" in result
    
    # Check that parameters have the expected length
    assert len(result["parameters"]) == 4
    assert len(result["parameter_names"]) == 4
    
    # Check that the fitted rates have the same length as the input rates
    assert len(result["fitted_rates"]) == len(rates)
    assert len(result["residuals"]) == len(rates)
    
    # Check that the prediction function works
    assert callable(result["predict_func"])
    predicted_rate = result["predict_func"](4.0)
    assert isinstance(predicted_rate, float)
    assert 0 <= predicted_rate <= 0.05  # Reasonable range for the test data
    
    # Check that R-squared is between 0 and 1
    assert 0 <= result["r_squared"] <= 1
    
    # Check that RMSE is positive
    assert result["rmse"] > 0
    
    # Check that the long rate is beta0
    beta0 = result["parameters"][0]
    assert result["long_rate"] == beta0


def test_nelson_siegel_initial_params():
    """Test Nelson-Siegel model with custom initial parameters."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    initial_params = [0.04, -0.03, -0.02, 2.0]  # [β₀, β₁, β₂, τ]
    
    result = nelson_siegel(maturities, rates, initial_params)
    
    # Check that the result contains the expected keys
    assert "parameters" in result
    
    # Check that the parameters are different from the initial parameters
    # (optimization should have changed them)
    assert result["parameters"] != initial_params
    
    # Check that the prediction function works with the new parameters
    assert callable(result["predict_func"])
    predicted_rate = result["predict_func"](4.0)
    assert isinstance(predicted_rate, float)


def test_nelson_siegel_fit_quality():
    """Test the quality of Nelson-Siegel model fit."""
    # Create synthetic data from a known Nelson-Siegel curve
    beta0, beta1, beta2, tau = 0.04, -0.02, -0.01, 1.5
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    
    # Calculate true rates using the Nelson-Siegel formula
    exp_term = np.exp(-maturities / tau)
    term1 = (1 - exp_term) / (maturities / tau)
    term2 = term1 - exp_term
    true_rates = beta0 + beta1 * term1 + beta2 * term2
    
    # Add small random noise
    np.random.seed(42)
    noisy_rates = true_rates + np.random.normal(0, 0.0005, len(maturities))
    
    # Fit the model
    result = nelson_siegel(maturities.tolist(), noisy_rates.tolist())
    
    # Check that the fitted parameters are close to the true parameters
    fitted_params = result["parameters"]
    assert abs(fitted_params[0] - beta0) < 0.01  # β₀
    assert abs(fitted_params[1] - beta1) < 0.01  # β₁
    assert abs(fitted_params[2] - beta2) < 0.01  # β₂
    assert abs(fitted_params[3] - tau) < 0.5     # τ (can be more variable)
    
    # Check that R-squared is close to 1 (good fit)
    assert result["r_squared"] > 0.95
    
    # Check that RMSE is small
    assert result["rmse"] < 0.001


def test_nelson_siegel_extrapolation():
    """Test Nelson-Siegel model extrapolation capabilities."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032]
    
    result = nelson_siegel(maturities, rates)
    
    # Extrapolate to longer maturities
    rate_20y = result["predict_func"](20)
    rate_30y = result["predict_func"](30)
    
    # Check that extrapolated rates are reasonable
    assert 0 <= rate_20y <= 0.05
    assert 0 <= rate_30y <= 0.05
    
    # Long-term rates should converge to beta0
    beta0 = result["parameters"][0]
    assert abs(rate_30y - beta0) < 0.005


def test_nelson_siegel_input_validation():
    """Test input validation for Nelson-Siegel model."""
    # Test mismatched lengths
    with pytest.raises(ValueError):
        nelson_siegel([1, 2, 3], [0.01, 0.02])
    
    # Test insufficient data points
    with pytest.raises(ValueError):
        nelson_siegel([1, 2, 3], [0.01, 0.02, 0.03])
    
    # Test non-positive maturities
    with pytest.raises(ValueError):
        nelson_siegel([0, 1, 2, 3, 4], [0.01, 0.02, 0.03, 0.04, 0.05])
    
    # Test negative rates
    with pytest.raises(ValueError):
        nelson_siegel([1, 2, 3, 4, 5], [-0.01, 0.02, 0.03, 0.04, 0.05])
    
    # Test invalid initial parameters
    with pytest.raises(ValueError):
        nelson_siegel([1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05], [0.03, -0.02])


def test_svensson_basic():
    """Test basic Svensson model fitting."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    result = svensson(maturities, rates)
    
    # Check that the result contains the expected keys
    assert "parameters" in result
    assert "parameter_names" in result
    assert "predict_func" in result
    assert "fitted_rates" in result
    assert "residuals" in result
    assert "r_squared" in result
    assert "rmse" in result
    assert "short_rate" in result
    assert "long_rate" in result
    
    # Check that parameters have the expected length
    assert len(result["parameters"]) == 6
    assert len(result["parameter_names"]) == 6
    
    # Check that the fitted rates have the same length as the input rates
    assert len(result["fitted_rates"]) == len(rates)
    assert len(result["residuals"]) == len(rates)
    
    # Check that the prediction function works
    assert callable(result["predict_func"])
    predicted_rate = result["predict_func"](4.0)
    assert isinstance(predicted_rate, float)
    assert 0 <= predicted_rate <= 0.05  # Reasonable range for the test data
    
    # Check that R-squared is between 0 and 1
    assert 0 <= result["r_squared"] <= 1
    
    # Check that RMSE is positive
    assert result["rmse"] > 0
    
    # Check that the long rate is beta0
    beta0 = result["parameters"][0]
    assert result["long_rate"] == beta0


def test_svensson_initial_params():
    """Test Svensson model with custom initial parameters."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    initial_params = [0.04, -0.03, -0.02, 0.01, 2.0, 5.0]  # [β₀, β₁, β₂, β₃, τ₁, τ₂]
    
    result = svensson(maturities, rates, initial_params)
    
    # Check that the result contains the expected keys
    assert "parameters" in result
    
    # Check that the parameters are different from the initial parameters
    # (optimization should have changed them)
    assert result["parameters"] != initial_params
    
    # Check that the prediction function works with the new parameters
    assert callable(result["predict_func"])
    predicted_rate = result["predict_func"](4.0)
    assert isinstance(predicted_rate, float)


def test_svensson_fit_quality():
    """Test the quality of Svensson model fit."""
    # Create synthetic data from a known Svensson curve
    beta0, beta1, beta2, beta3, tau1, tau2 = 0.04, -0.02, -0.01, 0.005, 1.5, 8.0
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    
    # Calculate true rates using the Svensson formula
    exp_term1 = np.exp(-maturities / tau1)
    term1 = (1 - exp_term1) / (maturities / tau1)
    term2 = term1 - exp_term1
    
    exp_term2 = np.exp(-maturities / tau2)
    term3 = (1 - exp_term2) / (maturities / tau2)
    term4 = term3 - exp_term2
    
    true_rates = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term4
    
    # Add small random noise
    np.random.seed(42)
    noisy_rates = true_rates + np.random.normal(0, 0.0005, len(maturities))
    
    # Fit the model
    result = svensson(maturities.tolist(), noisy_rates.tolist())
    
    # Check that R-squared is close to 1 (good fit)
    assert result["r_squared"] > 0.95
    
    # Check that RMSE is small
    assert result["rmse"] < 0.001


def test_svensson_vs_nelson_siegel():
    """Test that Svensson model provides better fit than Nelson-Siegel for complex curves."""
    # Create synthetic data with a complex shape that Svensson should fit better
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    
    # Create a curve with two humps - this is a more complex shape that
    # should favor the Svensson model which has an extra term for a second hump
    rates = np.array([0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.029, 0.028, 0.03, 0.032, 0.03, 0.028])
    
    # Create synthetic Svensson curve that Nelson-Siegel can't fit perfectly
    beta0, beta1, beta2, beta3, tau1, tau2 = 0.03, -0.02, -0.01, 0.01, 1.5, 10.0
    
    # Calculate true rates using the Svensson formula
    exp_term1 = np.exp(-maturities / tau1)
    term1 = (1 - exp_term1) / (maturities / tau1)
    term2 = term1 - exp_term1
    
    exp_term2 = np.exp(-maturities / tau2)
    term3 = (1 - exp_term2) / (maturities / tau2)
    term4 = term3 - exp_term2
    
    synthetic_rates = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term4
    
    # Add small random noise
    np.random.seed(42)
    noisy_rates = synthetic_rates + np.random.normal(0, 0.0001, len(maturities))
    
    # Fit both models
    ns_result = nelson_siegel(maturities.tolist(), noisy_rates.tolist())
    sv_result = svensson(maturities.tolist(), noisy_rates.tolist())
    
    # Svensson should have better R-squared and RMSE
    assert sv_result["r_squared"] >= ns_result["r_squared"]
    assert sv_result["rmse"] <= ns_result["rmse"]


def test_svensson_input_validation():
    """Test input validation for Svensson model."""
    # Test mismatched lengths
    with pytest.raises(ValueError):
        svensson([1, 2, 3], [0.01, 0.02])
    
    # Test insufficient data points
    with pytest.raises(ValueError):
        svensson([1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05])
    
    # Test non-positive maturities
    with pytest.raises(ValueError):
        svensson([0, 1, 2, 3, 4, 5, 6], [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    
    # Test negative rates
    with pytest.raises(ValueError):
        svensson([1, 2, 3, 4, 5, 6, 7], [-0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    
    # Test invalid initial parameters
    with pytest.raises(ValueError):
        svensson([1, 2, 3, 4, 5, 6, 7], [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], [0.03, -0.02])


def test_yield_curve_shapes():
    """Test that models can fit different yield curve shapes."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    
    # Normal (upward sloping) yield curve
    normal_rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    # Inverted (downward sloping) yield curve
    inverted_rates = [0.035, 0.033, 0.03, 0.028, 0.026, 0.024, 0.022, 0.02, 0.018, 0.017]
    
    # Humped yield curve
    humped_rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.029, 0.028, 0.027, 0.026]
    
    # Fit models to each curve shape
    normal_ns = nelson_siegel(maturities, normal_rates)
    inverted_ns = nelson_siegel(maturities, inverted_rates)
    humped_ns = nelson_siegel(maturities, humped_rates)
    
    normal_sv = svensson(maturities, normal_rates)
    inverted_sv = svensson(maturities, inverted_rates)
    humped_sv = svensson(maturities, humped_rates)
    
    # Check that all fits have reasonable R-squared values
    assert normal_ns["r_squared"] > 0.9
    assert inverted_ns["r_squared"] > 0.9
    assert humped_ns["r_squared"] > 0.9
    
    assert normal_sv["r_squared"] > 0.9
    assert inverted_sv["r_squared"] > 0.9
    assert humped_sv["r_squared"] > 0.9
    
    # Svensson should fit the humped curve better than Nelson-Siegel
    assert humped_sv["r_squared"] >= humped_ns["r_squared"]


def test_forward_rates():
    """Test calculation of forward rates from fitted models."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    # Fit models
    ns_result = nelson_siegel(maturities, rates)
    sv_result = svensson(maturities, rates)
    
    # Calculate 1-year forward rate starting in 2 years
    # f(2,3) = (3*r(3) - 2*r(2)) / (3-2)
    # where r(t) is the spot rate for maturity t
    
    # Using Nelson-Siegel
    r2 = ns_result["predict_func"](2)
    r3 = ns_result["predict_func"](3)
    forward_rate_ns = (3 * r3 - 2 * r2)
    
    # Using Svensson
    r2 = sv_result["predict_func"](2)
    r3 = sv_result["predict_func"](3)
    forward_rate_sv = (3 * r3 - 2 * r2)
    
    # Forward rates should be positive and reasonable
    assert forward_rate_ns > 0
    assert forward_rate_sv > 0
    assert 0 <= forward_rate_ns <= 0.05
    assert 0 <= forward_rate_sv <= 0.05


def test_parameter_interpretation():
    """Test interpretation of model parameters."""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    rates = [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.031, 0.032, 0.033, 0.034]
    
    # Fit models
    ns_result = nelson_siegel(maturities, rates)
    sv_result = svensson(maturities, rates)
    
    # Extract parameters
    ns_params = ns_result["parameters"]
    sv_params = sv_result["parameters"]
    
    # Nelson-Siegel parameters
    beta0_ns = ns_params[0]  # Long-term rate
    beta1_ns = ns_params[1]  # Short-term component
    beta2_ns = ns_params[2]  # Medium-term component
    tau_ns = ns_params[3]    # Decay factor
    
    # Svensson parameters
    beta0_sv = sv_params[0]  # Long-term rate
    beta1_sv = sv_params[1]  # Short-term component
    beta2_sv = sv_params[2]  # Medium-term component
    beta3_sv = sv_params[3]  # Second hump/trough component
    tau1_sv = sv_params[4]   # First decay factor
    tau2_sv = sv_params[5]   # Second decay factor
    
    # Check that long-term rates match
    assert ns_result["long_rate"] == beta0_ns
    assert sv_result["long_rate"] == beta0_sv
    
    # Check that short-term rates are influenced by beta1
    short_rate_ns = ns_result["predict_func"](0.1)
    assert abs(short_rate_ns - (beta0_ns + beta1_ns)) < 0.01
    
    # Check that tau values are positive
    assert tau_ns > 0
    assert tau1_sv > 0
    assert tau2_sv > 0 