import pytest
import numpy as np
from src.pypulate.kpi.business_kpi import (
    churn_rate, retention_rate, customer_lifetime_value, customer_acquisition_cost,
    monthly_recurring_revenue, annual_recurring_revenue, net_promoter_score,
    revenue_churn_rate, expansion_revenue_rate, ltv_cac_ratio, payback_period,
    customer_satisfaction_score, customer_effort_score, average_revenue_per_user,
    average_revenue_per_paying_user, conversion_rate, customer_engagement_score,
    daily_active_users_ratio, monthly_active_users_ratio, stickiness_ratio,
    gross_margin, burn_rate, runway, virality_coefficient, time_to_value,
    feature_adoption_rate, roi
)
from src.pypulate.dtypes.parray import Parray

class TestBusinessKPI:
    """Test class for business KPI functions."""
    
    def test_churn_rate_scalar(self):
        """Test churn_rate with scalar inputs."""
        # Basic test
        assert churn_rate(100, 90, 10) == 20.0
        
        # Edge case: zero customers at start
        assert churn_rate(0, 10, 10) == 0.0
        
        # Edge case: no churn
        assert churn_rate(100, 110, 10) == 0.0
        
        # Edge case: 100% churn
        assert churn_rate(100, 0, 0) == 100.0
    
    def test_churn_rate_array(self):
        """Test churn_rate with array inputs."""
        customers_start = np.array([100, 200, 300, 0])
        customers_end = np.array([90, 180, 300, 10])
        new_customers = np.array([10, 20, 30, 10])
        
        result = churn_rate(customers_start, customers_end, new_customers)
        expected = np.array([20.0, 20.0, 10.0, 0.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_retention_rate_scalar(self):
        """Test retention_rate with scalar inputs."""
        # Basic test
        assert retention_rate(100, 90, 10) == 80.0
        
        # Edge case: zero customers at start
        assert retention_rate(0, 10, 10) == 100.0
        
        # Edge case: full retention
        assert retention_rate(100, 110, 10) == 100.0
        
        # Edge case: zero retention
        assert retention_rate(100, 0, 0) == 0.0
    
    def test_retention_rate_array(self):
        """Test retention_rate with array inputs."""
        customers_start = np.array([100, 200, 300, 0])
        customers_end = np.array([90, 180, 300, 10])
        new_customers = np.array([10, 20, 30, 10])
        
        result = retention_rate(customers_start, customers_end, new_customers)
        expected = np.array([80.0, 80.0, 90.0, 100.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_customer_lifetime_value(self):
        """Test customer_lifetime_value function."""
        # Basic test
        clv = customer_lifetime_value(100, 70, 5, 10)
        assert round(clv, 2) == 466.67
        
        # Edge case: zero churn rate (infinite lifetime)
        clv = customer_lifetime_value(100, 70, 0, 10)
        assert clv > 0  # Should be a positive value based on discounted cash flow
        
        # Edge case: high churn rate (short lifetime)
        clv = customer_lifetime_value(100, 70, 50, 10)
        assert round(clv, 2) == 116.67  # Updated expected value
    
    def test_customer_acquisition_cost_scalar(self):
        """Test customer_acquisition_cost with scalar inputs."""
        # Basic test
        assert customer_acquisition_cost(5000, 3000, 100) == 80.0
        
        # Edge case: zero new customers
        assert customer_acquisition_cost(5000, 3000, 0) == float('inf')
        
        # Edge case: zero costs
        assert customer_acquisition_cost(0, 0, 100) == 0.0
    
    def test_customer_acquisition_cost_array(self):
        """Test customer_acquisition_cost with array inputs."""
        marketing_costs = np.array([5000, 6000, 7000, 0])
        sales_costs = np.array([3000, 4000, 5000, 0])
        new_customers = np.array([100, 200, 0, 50])
        
        result = customer_acquisition_cost(marketing_costs, sales_costs, new_customers)
        expected = np.array([80.0, 50.0, float('inf'), 0.0])
        
        # Test finite values
        np.testing.assert_array_almost_equal(
            result[~np.isinf(result)], 
            expected[~np.isinf(expected)]
        )
        # Test infinite values
        assert np.array_equal(np.isinf(result), np.isinf(expected))
    
    def test_monthly_recurring_revenue_scalar(self):
        """Test monthly_recurring_revenue with scalar inputs."""
        assert monthly_recurring_revenue(100, 50) == 5000.0
        assert monthly_recurring_revenue(0, 50) == 0.0
        assert monthly_recurring_revenue(100, 0) == 0.0
    
    def test_monthly_recurring_revenue_array(self):
        """Test monthly_recurring_revenue with array inputs."""
        paying_customers = np.array([100, 200, 300, 0])
        avg_revenue = np.array([50, 60, 70, 80])
        
        result = monthly_recurring_revenue(paying_customers, avg_revenue)
        expected = np.array([5000.0, 12000.0, 21000.0, 0.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_annual_recurring_revenue(self):
        """Test annual_recurring_revenue function."""
        assert annual_recurring_revenue(100, 50) == 60000.0
        assert annual_recurring_revenue(0, 50) == 0.0
        assert annual_recurring_revenue(100, 0) == 0.0
    
    def test_net_promoter_score_scalar(self):
        """Test net_promoter_score with scalar inputs."""
        # Basic test
        assert net_promoter_score(70, 10, 100) == 60.0
        
        # Edge case: zero respondents
        assert net_promoter_score(0, 0, 0) == 0.0
        
        # Edge case: all promoters
        assert net_promoter_score(100, 0, 100) == 100.0
        
        # Edge case: all detractors
        assert net_promoter_score(0, 100, 100) == -100.0
    
    def test_net_promoter_score_array(self):
        """Test net_promoter_score with array inputs."""
        promoters = np.array([70, 80, 0, 50])
        detractors = np.array([10, 20, 100, 50])
        total_respondents = np.array([100, 100, 100, 100])
        
        result = net_promoter_score(promoters, detractors, total_respondents)
        expected = np.array([60.0, 60.0, -100.0, 0.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_revenue_churn_rate_scalar(self):
        """Test revenue_churn_rate with scalar inputs."""
        # Basic test
        assert revenue_churn_rate(10000, 9500, 1000) == 15.0
        
        # Edge case: zero revenue at start
        assert revenue_churn_rate(0, 1000, 1000) == 0.0
        
        # Edge case: no churn
        assert revenue_churn_rate(10000, 11000, 1000) == 0.0
        
        # Edge case: 100% churn
        assert revenue_churn_rate(10000, 0, 0) == 100.0
    
    def test_revenue_churn_rate_array(self):
        """Test revenue_churn_rate with array inputs."""
        revenue_start = np.array([10000, 20000, 30000, 0])
        revenue_end = np.array([9500, 19000, 31000, 1000])
        new_revenue = np.array([1000, 2000, 3000, 1000])
        
        result = revenue_churn_rate(revenue_start, revenue_end, new_revenue)
        expected = np.array([15.0, 15.0, 6.67, 0.0])
        
        np.testing.assert_array_almost_equal(result, expected, decimal=2)
    
    def test_expansion_revenue_rate(self):
        """Test expansion_revenue_rate function."""
        # Basic test
        assert expansion_revenue_rate(1000, 500, 10000) == 15.0
        
        # Edge case: zero revenue at start
        assert expansion_revenue_rate(1000, 500, 0) == 0.0
        
        # Edge case: zero expansion revenue
        assert expansion_revenue_rate(0, 0, 10000) == 0.0
    
    def test_ltv_cac_ratio(self):
        """Test ltv_cac_ratio function."""
        # Basic test
        assert ltv_cac_ratio(1000, 200) == 5.0
        
        # Edge case: zero CAC
        assert ltv_cac_ratio(1000, 0) == float('inf')
        
        # Edge case: zero LTV
        assert ltv_cac_ratio(0, 200) == 0.0
    
    def test_payback_period(self):
        """Test payback_period function."""
        # Basic test
        assert round(payback_period(1000, 100, 70), 2) == 14.29
        
        # Edge case: zero monthly revenue or margin
        assert payback_period(1000, 0, 70) == float('inf')
        assert payback_period(1000, 100, 0) == float('inf')
        
        # Edge case: zero CAC
        assert payback_period(0, 100, 70) == 0.0
    
    def test_customer_satisfaction_score(self):
        """Test customer_satisfaction_score function."""
        # Basic test
        score = customer_satisfaction_score([4, 5, 3, 5, 4])
        assert abs(score - 84.0) < 0.01  # Using approximate equality
        
        # Edge case: empty ratings
        assert customer_satisfaction_score([]) == 0.0
        
        # Edge case: all max ratings
        assert customer_satisfaction_score([5, 5, 5, 5, 5]) == 100.0
        
        # Edge case: all min ratings
        assert customer_satisfaction_score([1, 1, 1, 1, 1]) == 20.0
        
        # Test with different max_rating
        score = customer_satisfaction_score([7, 8, 9, 10], max_rating=10)
        assert abs(score - 85.0) < 0.01  # Using approximate equality
    
    def test_customer_effort_score(self):
        """Test customer_effort_score function."""
        # Basic test
        assert customer_effort_score([2, 3, 1, 2, 4]) == 2.4
        
        # Edge case: empty ratings
        assert customer_effort_score([]) == 0.0
        
        # Edge case: all max ratings (worse)
        assert customer_effort_score([7, 7, 7, 7, 7]) == 7.0
        
        # Edge case: all min ratings (better)
        assert customer_effort_score([1, 1, 1, 1, 1]) == 1.0
        
        # Test with different max_rating
        assert customer_effort_score([2, 3, 4, 5], max_rating=5) == 3.5
    
    def test_average_revenue_per_user_scalar(self):
        """Test average_revenue_per_user with scalar inputs."""
        # Basic test
        assert average_revenue_per_user(10000, 500) == 20.0
        
        # Edge case: zero users
        assert average_revenue_per_user(10000, 0) == 0.0
        
        # Edge case: zero revenue
        assert average_revenue_per_user(0, 500) == 0.0
    
    def test_average_revenue_per_user_array(self):
        """Test average_revenue_per_user with array inputs."""
        total_revenue = np.array([10000, 20000, 30000, 0])
        total_users = np.array([500, 1000, 0, 100])
        
        result = average_revenue_per_user(total_revenue, total_users)
        expected = np.array([20.0, 20.0, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_average_revenue_per_paying_user(self):
        """Test average_revenue_per_paying_user function."""
        # Basic test
        assert average_revenue_per_paying_user(10000, 200) == 50.0
        
        # Edge case: zero paying users
        assert average_revenue_per_paying_user(10000, 0) == 0.0
        
        # Edge case: zero revenue
        assert average_revenue_per_paying_user(0, 200) == 0.0
    
    def test_conversion_rate_scalar(self):
        """Test conversion_rate with scalar inputs."""
        # Basic test
        assert conversion_rate(50, 1000) == 5.0
        
        # Edge case: zero visitors
        assert conversion_rate(50, 0) == 0.0
        
        # Edge case: zero conversions
        assert conversion_rate(0, 1000) == 0.0
        
        # Edge case: 100% conversion
        assert conversion_rate(1000, 1000) == 100.0
    
    def test_conversion_rate_array(self):
        """Test conversion_rate with array inputs."""
        conversions = np.array([50, 100, 0, 1000])
        total_visitors = np.array([1000, 1000, 1000, 1000])
        
        result = conversion_rate(conversions, total_visitors)
        expected = np.array([5.0, 10.0, 0.0, 100.0])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_customer_engagement_score(self):
        """Test customer_engagement_score function."""
        # Basic test
        assert customer_engagement_score(15, 30) == 50.0
        
        # Edge case: zero total days
        assert customer_engagement_score(15, 0) == 0.0
        
        # Edge case: zero active days
        assert customer_engagement_score(0, 30) == 0.0
        
        # Edge case: 100% engagement
        assert customer_engagement_score(30, 30) == 100.0
    
    def test_daily_active_users_ratio(self):
        """Test daily_active_users_ratio function."""
        # Basic test
        assert daily_active_users_ratio(500, 2000) == 25.0
        
        # Edge case: zero total users
        assert daily_active_users_ratio(500, 0) == 0.0
        
        # Edge case: zero active users
        assert daily_active_users_ratio(0, 2000) == 0.0
        
        # Edge case: 100% active users
        assert daily_active_users_ratio(2000, 2000) == 100.0
    
    def test_monthly_active_users_ratio(self):
        """Test monthly_active_users_ratio function."""
        # Basic test
        assert monthly_active_users_ratio(1500, 2000) == 75.0
        
        # Edge case: zero total users
        assert monthly_active_users_ratio(1500, 0) == 0.0
        
        # Edge case: zero active users
        assert monthly_active_users_ratio(0, 2000) == 0.0
        
        # Edge case: 100% active users
        assert monthly_active_users_ratio(2000, 2000) == 100.0
    
    def test_stickiness_ratio(self):
        """Test stickiness_ratio function."""
        # Basic test
        assert round(stickiness_ratio(500, 1500), 2) == 33.33
        
        # Edge case: zero monthly active users
        assert stickiness_ratio(500, 0) == 0.0
        
        # Edge case: zero daily active users
        assert stickiness_ratio(0, 1500) == 0.0
        
        # Edge case: DAU = MAU
        assert stickiness_ratio(1500, 1500) == 100.0
    
    def test_gross_margin(self):
        """Test gross_margin function."""
        # Basic test
        assert gross_margin(10000, 3000) == 70.0
        
        # Edge case: zero revenue
        assert gross_margin(0, 3000) == 0.0
        
        # Edge case: zero COGS
        assert gross_margin(10000, 0) == 100.0
        
        # Edge case: COGS > revenue (negative margin)
        assert gross_margin(10000, 15000) == -50.0
    
    def test_burn_rate(self):
        """Test burn_rate function."""
        # Basic test
        assert burn_rate(100000, 70000, 6) == 5000.0
        
        # Edge case: zero months
        assert burn_rate(100000, 70000, 0) == 0.0
        
        # Edge case: no burn (ending capital > starting capital)
        burn = burn_rate(100000, 120000, 6)
        assert abs(burn + 3333.33) < 0.01  # Using approximate equality
        
        # Edge case: complete burn
        burn = burn_rate(100000, 0, 6)
        assert abs(burn - 16666.67) < 0.01  # Using approximate equality
    
    def test_runway(self):
        """Test runway function."""
        # Basic test
        assert runway(100000, 5000) == 20.0
        
        # Edge case: zero burn rate
        assert runway(100000, 0) == float('inf')
        
        # Edge case: zero capital
        assert runway(0, 5000) == 0.0
        
        # Edge case: negative burn rate (increasing capital)
        assert runway(100000, -5000) == -20.0
    
    def test_virality_coefficient(self):
        """Test virality_coefficient function."""
        # Basic test
        assert virality_coefficient(100, 500, 1000) == 0.1
        
        # Edge case: zero users or invites
        assert virality_coefficient(100, 0, 1000) == 0.0
        assert virality_coefficient(100, 500, 0) == 0.0
        
        # Edge case: high virality (k > 1)
        assert virality_coefficient(600, 1000, 1000) == 0.6
    
    def test_time_to_value(self):
        """Test time_to_value function."""
        # Basic test
        assert time_to_value(2, 3, 5) == 10.0
        
        # Edge case: zero times
        assert time_to_value(0, 0, 0) == 0.0
        
        # Edge case: only one component
        assert time_to_value(5, 0, 0) == 5.0
    
    def test_feature_adoption_rate(self):
        """Test feature_adoption_rate function."""
        # Basic test
        assert feature_adoption_rate(300, 1000) == 30.0
        
        # Edge case: zero users
        assert feature_adoption_rate(300, 0) == 0.0
        
        # Edge case: zero adopters
        assert feature_adoption_rate(0, 1000) == 0.0
        
        # Edge case: 100% adoption
        assert feature_adoption_rate(1000, 1000) == 100.0
    
    def test_roi_scalar(self):
        """Test roi with scalar inputs."""
        # Basic test
        assert roi(150, 100) == 50.0
        
        # Edge case: zero cost
        assert roi(150, 0) == 0.0
        
        # Edge case: zero revenue
        assert roi(0, 100) == -100.0
        
        # Edge case: revenue = cost (break-even)
        assert roi(100, 100) == 0.0
        
        # Edge case: negative ROI
        assert roi(50, 100) == -50.0
    
    def test_roi_array(self):
        """Test roi with array inputs."""
        revenue = np.array([150, 200, 50, 100])
        costs = np.array([100, 120, 100, 100])
        
        result = roi(revenue, costs)
        expected = np.array([50.0, 66.67, -50.0, 0.0])
        
        np.testing.assert_array_almost_equal(result, expected, decimal=2)
    
    def test_with_parray(self):
        """Test functions with Parray inputs."""
        # Create Parray inputs
        customers_start = Parray([100, 200, 300, 0])
        customers_end = Parray([90, 180, 300, 10])
        new_customers = Parray([10, 20, 30, 10])
        
        # Test churn_rate
        result = churn_rate(customers_start, customers_end, new_customers)
        expected = np.array([20.0, 20.0, 10.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test retention_rate
        result = retention_rate(customers_start, customers_end, new_customers)
        expected = np.array([80.0, 80.0, 90.0, 100.0])
        np.testing.assert_array_almost_equal(result, expected) 