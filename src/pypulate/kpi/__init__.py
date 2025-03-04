"""
Key Performance Indicators (KPI) module.

This module provides functions for calculating various business and financial KPIs.
"""

from .business_kpi import (
    churn_rate, retention_rate,
    customer_lifetime_value, customer_acquisition_cost, ltv_cac_ratio, payback_period,
    monthly_recurring_revenue, annual_recurring_revenue,
    revenue_churn_rate, expansion_revenue_rate,
    customer_satisfaction_score, customer_effort_score, net_promoter_score,
    daily_active_users_ratio, monthly_active_users_ratio, stickiness_ratio,
    burn_rate, runway, gross_margin,
    conversion_rate, virality_coefficient, feature_adoption_rate, roi,
    average_revenue_per_user, average_revenue_per_paying_user, customer_engagement_score, time_to_value
)


__all__ = [
    'churn_rate', 'retention_rate',
    'customer_lifetime_value', 'customer_acquisition_cost', 'ltv_cac_ratio', 'payback_period',
    'monthly_recurring_revenue', 'annual_recurring_revenue',
    'revenue_churn_rate', 'expansion_revenue_rate',
    'customer_satisfaction_score', 'customer_effort_score', 'net_promoter_score',
    'daily_active_users_ratio', 'monthly_active_users_ratio', 'stickiness_ratio',
    'burn_rate', 'runway', 'gross_margin',
    'conversion_rate', 'virality_coefficient', 'feature_adoption_rate', 'roi',
    'average_revenue_per_user', 'average_revenue_per_paying_user', 'customer_engagement_score', 'time_to_value'
]
