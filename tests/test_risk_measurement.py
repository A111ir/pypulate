import unittest
import numpy as np
from pypulate.portfolio.risk_measurement import (
    standard_deviation,
    semi_standard_deviation,
    tracking_error,
    capm_beta,
    value_at_risk,
    covariance_matrix,
    correlation_matrix,
    conditional_value_at_risk,
    drawdown
)


class TestRiskMeasurement(unittest.TestCase):
    def setUp(self):
        # Sample return data as lists
        self.returns_list = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02, -0.01]
        self.benchmark_returns_list = [0.005, -0.01, 0.02, -0.005, 0.015, -0.02, 0.01, 0.015, -0.005]
        self.market_returns_list = [0.008, -0.015, 0.025, -0.008, 0.018, -0.025, 0.012, 0.018, -0.008]
        
        # Same data as numpy arrays
        self.returns_array = np.array(self.returns_list)
        self.benchmark_returns_array = np.array(self.benchmark_returns_list)
        self.market_returns_array = np.array(self.market_returns_list)
        
        # Multi-asset return data as lists
        self.multi_asset_returns_list = [
            [0.01, 0.005, 0.008],
            [-0.02, -0.01, -0.015],
            [0.03, 0.02, 0.025],
            [-0.01, -0.005, -0.008],
            [0.02, 0.015, 0.018],
            [-0.03, -0.02, -0.025],
            [0.01, 0.01, 0.012],
            [0.02, 0.015, 0.018],
            [-0.01, -0.005, -0.008]
        ]
        
        # Same multi-asset data as numpy array
        self.multi_asset_returns_array = np.array(self.multi_asset_returns_list)

    def test_standard_deviation(self):
        # Test with list input
        sd_list = standard_deviation(self.returns_list)
        self.assertIsInstance(sd_list, float)
        
        # Test with numpy array input
        sd_array = standard_deviation(self.returns_array)
        self.assertIsInstance(sd_array, float)
        
        # Results should be the same regardless of input type
        self.assertAlmostEqual(sd_list, sd_array)
        
        # Test annualization
        sd_annual = standard_deviation(self.returns_list, annualize=True, periods_per_year=252)
        self.assertGreater(sd_annual, sd_list)  # Annualized value should be larger

    def test_semi_standard_deviation(self):
        # Test with list input
        semi_sd_list = semi_standard_deviation(self.returns_list)
        self.assertIsInstance(semi_sd_list, float)
        
        # Test with numpy array input
        semi_sd_array = semi_standard_deviation(self.returns_array)
        self.assertIsInstance(semi_sd_array, float)
        
        # Results should be the same regardless of input type
        self.assertAlmostEqual(semi_sd_list, semi_sd_array)
        
        # Test with different threshold
        semi_sd_threshold = semi_standard_deviation(self.returns_list, threshold=0.01)
        self.assertIsInstance(semi_sd_threshold, float)

    def test_tracking_error(self):
        # Test with list input
        te_list = tracking_error(self.returns_list, self.benchmark_returns_list)
        self.assertIsInstance(te_list, float)
        
        # Test with numpy array input
        te_array = tracking_error(self.returns_array, self.benchmark_returns_array)
        self.assertIsInstance(te_array, float)
        
        # Results should be the same regardless of input type
        self.assertAlmostEqual(te_list, te_array)
        
        # Test annualization
        te_annual = tracking_error(self.returns_list, self.benchmark_returns_list, annualize=True)
        self.assertGreater(te_annual, te_list)  # Annualized value should be larger

    def test_capm_beta(self):
        # Test with list input
        beta_list = capm_beta(self.returns_list, self.market_returns_list)
        self.assertIsInstance(beta_list, float)
        
        # Test with numpy array input
        beta_array = capm_beta(self.returns_array, self.market_returns_array)
        self.assertIsInstance(beta_array, float)
        
        # Results should be the same regardless of input type
        self.assertAlmostEqual(beta_list, beta_array)

    def test_value_at_risk(self):
        # Test with list input - historical method
        var_hist_list = value_at_risk(self.returns_list, method='historical')
        self.assertIsInstance(var_hist_list, float)
        
        # Test with numpy array input - historical method
        var_hist_array = value_at_risk(self.returns_array, method='historical')
        self.assertIsInstance(var_hist_array, float)
        
        # Results should be the same regardless of input type
        self.assertAlmostEqual(var_hist_list, var_hist_array)
        
        # Test parametric method
        var_param = value_at_risk(self.returns_list, method='parametric')
        self.assertIsInstance(var_param, float)
        
        # Test monte carlo method
        var_mc = value_at_risk(self.returns_list, method='monte_carlo')
        self.assertIsInstance(var_mc, float)
        
        # Test different confidence level
        var_conf = value_at_risk(self.returns_list, confidence_level=0.99)
        self.assertIsInstance(var_conf, float)
        self.assertGreater(var_conf, var_hist_list)  # Higher confidence should give higher VaR

    def test_covariance_matrix(self):
        # Test with list input, numpy output
        cov_list_np = covariance_matrix(self.multi_asset_returns_list)
        self.assertIsInstance(cov_list_np, np.ndarray)
        self.assertEqual(cov_list_np.shape, (3, 3))  # 3x3 covariance matrix for 3 assets
        
        # Test with numpy array input, numpy output
        cov_np_np = covariance_matrix(self.multi_asset_returns_array)
        self.assertIsInstance(cov_np_np, np.ndarray)
        

    def test_correlation_matrix(self):
        # Test with list input, numpy output
        corr_list_np = correlation_matrix(self.multi_asset_returns_list)
        self.assertIsInstance(corr_list_np, np.ndarray)
        self.assertEqual(corr_list_np.shape, (3, 3))  # 3x3 correlation matrix for 3 assets

        
        # Test with numpy array input, numpy output
        corr_np_np = correlation_matrix(self.multi_asset_returns_array)
        self.assertIsInstance(corr_np_np, np.ndarray)
        

    def test_conditional_value_at_risk(self):
        # Test with list input - historical method
        cvar_hist_list = conditional_value_at_risk(self.returns_list, method='historical')
        self.assertIsInstance(cvar_hist_list, float)
        
        # Test with numpy array input - historical method
        cvar_hist_array = conditional_value_at_risk(self.returns_array, method='historical')
        self.assertIsInstance(cvar_hist_array, float)
        
        # Results should be the same regardless of input type
        self.assertAlmostEqual(cvar_hist_list, cvar_hist_array)
        
        # Test parametric method
        cvar_param = conditional_value_at_risk(self.returns_list, method='parametric')
        self.assertIsInstance(cvar_param, float)
        
        # Test different confidence level
        cvar_conf = conditional_value_at_risk(self.returns_list, confidence_level=0.99)
        self.assertIsInstance(cvar_conf, float)
        
        # CVaR should be greater than or equal to VaR
        var_hist = value_at_risk(self.returns_list, method='historical')
        self.assertGreaterEqual(cvar_hist_list, var_hist)

    def test_drawdown(self):
        # Test with list input, numpy output
        dd_list_np, max_dd_list, start_idx_list, end_idx_list = drawdown(self.returns_list, as_list=False)
        self.assertIsInstance(dd_list_np, np.ndarray)
        self.assertIsInstance(max_dd_list, float)
        self.assertIsInstance(start_idx_list, int)
        self.assertIsInstance(end_idx_list, int)
        
        # Test with list input, list output
        dd_list_list, max_dd_list2, start_idx_list2, end_idx_list2 = drawdown(self.returns_list, as_list=True)
        self.assertIsInstance(dd_list_list, list)
        self.assertIsInstance(max_dd_list2, float)
        self.assertIsInstance(start_idx_list2, int)
        self.assertIsInstance(end_idx_list2, int)
        
        # Test with numpy array input, numpy output
        dd_np_np, max_dd_np, start_idx_np, end_idx_np = drawdown(self.returns_array)
        self.assertIsInstance(dd_np_np, np.ndarray)
        
        # Test with numpy array input, list output
        dd_np_list, max_dd_np2, start_idx_np2, end_idx_np2 = drawdown(self.returns_array, as_list=True)
        self.assertIsInstance(dd_np_list, list)
        
        # Results should be the same regardless of input/output type
        self.assertAlmostEqual(max_dd_list, max_dd_np)
        self.assertEqual(start_idx_list, start_idx_np)
        self.assertEqual(end_idx_list, end_idx_np)
        
        # Maximum drawdown should be positive
        self.assertGreaterEqual(max_dd_list, 0)


if __name__ == '__main__':
    unittest.main() 