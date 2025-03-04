"""
KPI Module

This module provides a class for calculating various business metrics
commonly used in SaaS and subscription-based businesses.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict
from ..kpi.business_kpi import (
    churn_rate, retention_rate, customer_lifetime_value,
    customer_acquisition_cost, monthly_recurring_revenue,
    annual_recurring_revenue, net_promoter_score,
    revenue_churn_rate, expansion_revenue_rate,
    ltv_cac_ratio, payback_period, customer_satisfaction_score,
    customer_effort_score, average_revenue_per_user,
    average_revenue_per_paying_user, conversion_rate,
    customer_engagement_score, daily_active_users_ratio,
    monthly_active_users_ratio, stickiness_ratio,
    gross_margin, burn_rate, runway, virality_coefficient,
    time_to_value, feature_adoption_rate, roi
)


class KPI:
    """
    A class for calculating various business KPIs.
    
    This class provides methods for calculating common business metrics
    used in SaaS and subscription-based businesses and maintains state
    to assess overall business health.
    
    Examples
    --------
    >>> from pypulate.dtypes import KPI
    >>> kpi = KPI()
    >>> churn = kpi.churn_rate(100, 90, 10)
    >>> retention = kpi.retention_rate(100, 90, 10)
    >>> clv = kpi.customer_lifetime_value(100, 70, 5)
    >>> health = kpi.health
    """
    
    def __init__(self):
        """Initialize the KPI class with empty state."""
        self._state = {
            'churn_rate': None,
            'retention_rate': None,
            'customer_lifetime_value': None,
            'customer_acquisition_cost': None,
            'monthly_recurring_revenue': None,
            'annual_recurring_revenue': None,
            'net_promoter_score': None,
            'revenue_churn_rate': None,
            'expansion_revenue_rate': None,
            'ltv_cac_ratio': None,
            'payback_period': None,
            'customer_satisfaction_score': None,
            'customer_effort_score': None,
            'average_revenue_per_user': None,
            'average_revenue_per_paying_user': None,
            'conversion_rate': None,
            'customer_engagement_score': None,
            'daily_active_users_ratio': None,
            'monthly_active_users_ratio': None,
            'stickiness_ratio': None,
            'gross_margin': None,
            'burn_rate': None,
            'runway': None,
            'virality_coefficient': None,
            'time_to_value': None,
            'feature_adoption_rate': None,
            'roi': None
        }
    
    @property
    def health(self) -> Dict[str, Union[float, str, Dict[str, Dict[str, Union[float, str]]]]]:
        """
        Calculate and return the overall health of the business based on stored KPIs.
        
        Returns
        -------
        dict
            Dictionary containing health score and individual component scores
        """
        health_score = 0.0
        components = {}
        
        if self._state['churn_rate'] is not None:
            churn_score = max(0, min(100, 100 - self._state['churn_rate']))
            health_score += churn_score * 0.10
            components['churn_rate'] = {
                'score': churn_score,
                'status': 'Excellent' if churn_score >= 95 else 'Good' if churn_score >= 85 else 'Fair' if churn_score >= 70 else 'Poor' if churn_score >= 50 else 'Critical'
            }
            
        if self._state['retention_rate'] is not None:
            retention_score = self._state['retention_rate']
            health_score += retention_score * 0.10
            components['retention_rate'] = {
                'score': retention_score,
                'status': 'Excellent' if retention_score >= 95 else 'Good' if retention_score >= 85 else 'Fair' if retention_score >= 70 else 'Poor' if retention_score >= 50 else 'Critical'
            }

        if self._state['ltv_cac_ratio'] is not None:
            ltv_cac_score = min(100, self._state['ltv_cac_ratio'] * 20)
            health_score += ltv_cac_score * 0.10
            components['ltv_cac_ratio'] = {
                'score': ltv_cac_score,
                'status': 'Excellent' if ltv_cac_score >= 80 else 'Good' if ltv_cac_score >= 60 else 'Fair' if ltv_cac_score >= 40 else 'Poor' if ltv_cac_score >= 20 else 'Critical'
            }

        if self._state['gross_margin'] is not None:
            margin_score = self._state['gross_margin']
            health_score += margin_score * 0.10
            components['gross_margin'] = {
                'score': margin_score,
                'status': 'Excellent' if margin_score >= 80 else 'Good' if margin_score >= 70 else 'Fair' if margin_score >= 50 else 'Poor' if margin_score >= 30 else 'Critical'
            }

        if self._state['net_promoter_score'] is not None:
            nps_score = max(0, min(100, self._state['net_promoter_score'] + 50))
            health_score += nps_score * 0.10
            components['net_promoter_score'] = {
                'score': nps_score,
                'status': 'Excellent' if nps_score >= 60 else 'Good' if nps_score >= 50 else 'Fair' if nps_score >= 30 else 'Poor' if nps_score >= 20 else 'Critical'
            }

        if self._state['customer_satisfaction_score'] is not None:
            csat_score = self._state['customer_satisfaction_score']
            health_score += csat_score * 0.10
            components['customer_satisfaction_score'] = {
                'score': csat_score,
                'status': 'Excellent' if csat_score >= 90 else 'Good' if csat_score >= 85 else 'Fair' if csat_score >= 70 else 'Poor' if csat_score >= 50 else 'Critical'
            }

        if self._state['expansion_revenue_rate'] is not None:
            expansion_score = min(100, max(0, self._state['expansion_revenue_rate'] * 2))
            health_score += expansion_score * 0.10
            components['expansion_revenue_rate'] = {
                'score': expansion_score,
                'status': 'Excellent' if expansion_score >= 30 else 'Good' if expansion_score >= 20 else 'Fair' if expansion_score >= 10 else 'Poor' if expansion_score >= 5 else 'Critical'
            }

        if self._state['stickiness_ratio'] is not None:
            stickiness_score = self._state['stickiness_ratio']
            health_score += stickiness_score * 0.10
            components['stickiness_ratio'] = {
                'score': stickiness_score,
                'status': 'Excellent' if stickiness_score >= 70 else 'Good' if stickiness_score >= 50 else 'Fair' if stickiness_score >= 30 else 'Poor' if stickiness_score >= 20 else 'Critical'
            }

        if self._state['runway'] is not None:
            runway_score = min(100, max(0, self._state['runway'] * 5))
            health_score += runway_score * 0.10
            components['runway'] = {
                'score': runway_score,
                'status': 'Excellent' if runway_score >= 18 else 'Good' if runway_score >= 12 else 'Fair' if runway_score >= 6 else 'Poor' if runway_score >= 3 else 'Critical'
            }

        if self._state['roi'] is not None:
            roi_score = min(100, max(0, self._state['roi']))
            health_score += roi_score * 0.10
            components['roi'] = {
                'score': roi_score,
                'status': 'Excellent' if roi_score >= 50 else 'Good' if roi_score >= 30 else 'Fair' if roi_score >= 15 else 'Poor' if roi_score >= 5 else 'Critical'
            }

        return {
            'overall_score': health_score,
            'status': 'Excellent' if health_score >= 90 else 'Good' if health_score >= 75 else 'Fair' if health_score >= 60 else 'Poor' if health_score >= 45 else 'Critical',
            'components': components
        }
    
    def churn_rate(
        self,
        customers_start: Union[int, float, np.ndarray, list],
        customers_end: Union[int, float, np.ndarray, list],
        new_customers: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate customer churn rate.
        
        Churn rate is the percentage of customers who stop using your product or service
        during a given time period.
        
        Parameters
        ----------
        customers_start : int, float, array-like
            Number of customers at the start of the period
        customers_end : int, float, array-like
            Number of customers at the end of the period
        new_customers : int, float, array-like
            Number of new customers acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Churn rate as a percentage
        """
        result = churn_rate(customers_start, customers_end, new_customers)
        if isinstance(result, (int, float)):
            self._state['churn_rate'] = result
        return result
    
    def retention_rate(
        self,
        customers_start: Union[int, float, np.ndarray, list],
        customers_end: Union[int, float, np.ndarray, list],
        new_customers: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate customer retention rate.
        
        Retention rate is the percentage of customers who remain with your product or service
        over a given time period.
        
        Parameters
        ----------
        customers_start : int, float, array-like
            Number of customers at the start of the period
        customers_end : int, float, array-like
            Number of customers at the end of the period
        new_customers : int, float, array-like
            Number of new customers acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Retention rate as a percentage
        """
        result = retention_rate(customers_start, customers_end, new_customers)
        if isinstance(result, (int, float)):
            self._state['retention_rate'] = result
        return result
    
    def customer_lifetime_value(
        self,
        avg_revenue_per_customer: Union[int, float],
        gross_margin: Union[int, float],
        churn_rate_value: Union[int, float],
        discount_rate: Union[int, float] = 10.0
    ) -> float:
        """
        Calculate Customer Lifetime Value (CLV).
        
        CLV is the total worth to a business of a customer over the whole period of their relationship.
        
        Parameters
        ----------
        avg_revenue_per_customer : int or float
            Average revenue per customer per period (e.g., monthly)
        gross_margin : int or float
            Gross margin percentage (0-100)
        churn_rate_value : int or float
            Churn rate percentage (0-100)
        discount_rate : int or float, default 10.0
            Annual discount rate for future cash flows (0-100)
            
        Returns
        -------
        float
            Customer Lifetime Value
        """
        result = customer_lifetime_value(
            avg_revenue_per_customer, gross_margin, churn_rate_value, discount_rate
        )
        self._state['customer_lifetime_value'] = result
        return result
    
    def customer_acquisition_cost(
        self,
        marketing_costs: Union[int, float, np.ndarray, list],
        sales_costs: Union[int, float, np.ndarray, list],
        new_customers: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate Customer Acquisition Cost (CAC).
        
        CAC is the cost of convincing a potential customer to buy a product or service.
        
        Parameters
        ----------
        marketing_costs : int, float, array-like
            Total marketing costs for the period
        sales_costs : int, float, array-like
            Total sales costs for the period
        new_customers : int, float, array-like
            Number of new customers acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Customer Acquisition Cost
        """
        result = customer_acquisition_cost(marketing_costs, sales_costs, new_customers)
        if isinstance(result, (int, float)):
            self._state['customer_acquisition_cost'] = result
        return result
    
    def monthly_recurring_revenue(
        self,
        paying_customers: Union[int, float, np.ndarray, list],
        avg_revenue_per_customer: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate Monthly Recurring Revenue (MRR).
        
        MRR is the predictable total revenue generated by all the active subscriptions in a month.
        
        Parameters
        ----------
        paying_customers : int, float, array-like
            Number of paying customers
        avg_revenue_per_customer : int, float, array-like
            Average revenue per customer per month
            
        Returns
        -------
        float or numpy.ndarray
            Monthly Recurring Revenue
        """
        result = monthly_recurring_revenue(paying_customers, avg_revenue_per_customer)
        if isinstance(result, (int, float)):
            self._state['monthly_recurring_revenue'] = result
        return result
    
    def annual_recurring_revenue(
        self,
        paying_customers: Union[int, float],
        avg_revenue_per_customer: Union[int, float]
    ) -> float:
        """
        Calculate Annual Recurring Revenue (ARR).
        
        ARR is the value of the recurring revenue of a business's term subscriptions normalized for a single calendar year.
        
        Parameters
        ----------
        paying_customers : int or float
            Number of paying customers
        avg_revenue_per_customer : int or float
            Average revenue per customer per month
            
        Returns
        -------
        float
            Annual Recurring Revenue
        """
        result = annual_recurring_revenue(paying_customers, avg_revenue_per_customer)
        self._state['annual_recurring_revenue'] = result
        return result
    
    def net_promoter_score(
        self,
        promoters: Union[int, float, np.ndarray, list],
        detractors: Union[int, float, np.ndarray, list],
        total_respondents: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate Net Promoter Score (NPS).
        
        NPS measures customer experience and predicts business growth.
        
        Parameters
        ----------
        promoters : int, float, array-like
            Number of promoters (customers who rated 9-10)
        detractors : int, float, array-like
            Number of detractors (customers who rated 0-6)
        total_respondents : int, float, array-like
            Total number of survey respondents
            
        Returns
        -------
        float or numpy.ndarray
            Net Promoter Score (ranges from -100 to 100)
        """
        result = net_promoter_score(promoters, detractors, total_respondents)
        if isinstance(result, (int, float)):
            self._state['net_promoter_score'] = result
        return result
    
    def revenue_churn_rate(
        self,
        revenue_start: Union[int, float, np.ndarray, list],
        revenue_end: Union[int, float, np.ndarray, list],
        new_revenue: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate Revenue Churn Rate.
        
        Revenue Churn Rate is the percentage of revenue lost from existing customers in a given period.
        
        Parameters
        ----------
        revenue_start : int, float, array-like
            Revenue at the start of the period
        revenue_end : int, float, array-like
            Revenue at the end of the period
        new_revenue : int, float, array-like
            New revenue acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Revenue Churn Rate as a percentage
        """
        result = revenue_churn_rate(revenue_start, revenue_end, new_revenue)
        if isinstance(result, (int, float)):
            self._state['revenue_churn_rate'] = result
        return result
    
    def expansion_revenue_rate(
        self,
        upsell_revenue: Union[int, float],
        cross_sell_revenue: Union[int, float],
        revenue_start: Union[int, float]
    ) -> float:
        """
        Calculate Expansion Revenue Rate.
        
        Expansion Revenue Rate is the percentage of additional revenue generated from existing customers.
        
        Parameters
        ----------
        upsell_revenue : int or float
            Revenue from upselling to existing customers
        cross_sell_revenue : int or float
            Revenue from cross-selling to existing customers
        revenue_start : int or float
            Revenue at the start of the period
            
        Returns
        -------
        float
            Expansion Revenue Rate as a percentage
        """
        result = expansion_revenue_rate(upsell_revenue, cross_sell_revenue, revenue_start)
        self._state['expansion_revenue_rate'] = result
        return result
    
    def ltv_cac_ratio(
        self,
        ltv: Union[int, float],
        cac: Union[int, float]
    ) -> float:
        """
        Calculate LTV:CAC Ratio.
        
        LTV:CAC Ratio is a metric that compares the lifetime value of a customer to the cost of acquiring that customer.
        
        Parameters
        ----------
        ltv : int or float
            Customer Lifetime Value
        cac : int or float
            Customer Acquisition Cost
            
        Returns
        -------
        float
            LTV:CAC Ratio
        """
        result = ltv_cac_ratio(ltv, cac)
        self._state['ltv_cac_ratio'] = result
        return result
    
    def payback_period(
        self,
        cac: Union[int, float],
        avg_monthly_revenue: Union[int, float],
        gross_margin: Union[int, float]
    ) -> float:
        """
        Calculate CAC Payback Period in months.
        
        CAC Payback Period is the number of months it takes to recover the cost of acquiring a customer.
        
        Parameters
        ----------
        cac : int or float
            Customer Acquisition Cost
        avg_monthly_revenue : int or float
            Average monthly revenue per customer
        gross_margin : int or float
            Gross margin percentage (0-100)
            
        Returns
        -------
        float
            CAC Payback Period in months
        """
        result = payback_period(cac, avg_monthly_revenue, gross_margin)
        self._state['payback_period'] = result
        return result
    
    def customer_satisfaction_score(
        self,
        satisfaction_ratings: Union[np.ndarray, list],
        max_rating: Union[int, float] = 5
    ) -> float:
        """
        Calculate Customer Satisfaction Score (CSAT).
        
        CSAT measures how satisfied customers are with a product, service, or interaction.
        
        Parameters
        ----------
        satisfaction_ratings : array-like
            Array of customer satisfaction ratings
        max_rating : int or float, default 5
            Maximum possible rating value
            
        Returns
        -------
        float
            Customer Satisfaction Score as a percentage
        """
        result = customer_satisfaction_score(satisfaction_ratings, max_rating)
        self._state['customer_satisfaction_score'] = result
        return result
    
    def customer_effort_score(
        self,
        effort_ratings: Union[np.ndarray, list],
        max_rating: Union[int, float] = 7
    ) -> float:
        """
        Calculate Customer Effort Score (CES).
        
        CES measures how much effort a customer has to exert to use a product or service.
        Lower scores are better.
        
        Parameters
        ----------
        effort_ratings : array-like
            Array of customer effort ratings
        max_rating : int or float, default 7
            Maximum possible rating value
            
        Returns
        -------
        float
            Customer Effort Score (average)
        """
        result = customer_effort_score(effort_ratings, max_rating)
        self._state['customer_effort_score'] = result
        return result
    
    def average_revenue_per_user(
        self,
        total_revenue: Union[int, float, np.ndarray, list],
        total_users: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate Average Revenue Per User (ARPU).
        
        ARPU measures the average revenue generated per user or customer.
        
        Parameters
        ----------
        total_revenue : int, float, array-like
            Total revenue for the period
        total_users : int, float, array-like
            Total number of users or customers
            
        Returns
        -------
        float or numpy.ndarray
            Average Revenue Per User
        """
        result = average_revenue_per_user(total_revenue, total_users)
        if isinstance(result, (int, float)):
            self._state['average_revenue_per_user'] = result
        return result
    
    def average_revenue_per_paying_user(
        self,
        total_revenue: Union[int, float],
        paying_users: Union[int, float]
    ) -> float:
        """
        Calculate Average Revenue Per Paying User (ARPPU).
        
        ARPPU measures the average revenue generated per paying user or customer.
        
        Parameters
        ----------
        total_revenue : int or float
            Total revenue for the period
        paying_users : int or float
            Number of paying users or customers
            
        Returns
        -------
        float
            Average Revenue Per Paying User
        """
        result = average_revenue_per_paying_user(total_revenue, paying_users)
        self._state['average_revenue_per_paying_user'] = result
        return result
    
    def conversion_rate(
        self,
        conversions: Union[int, float, np.ndarray, list],
        total_visitors: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate Conversion Rate.
        
        Conversion Rate is the percentage of visitors who take a desired action.
        
        Parameters
        ----------
        conversions : int, float, array-like
            Number of conversions (desired actions taken)
        total_visitors : int, float, array-like
            Total number of visitors or users
            
        Returns
        -------
        float or numpy.ndarray
            Conversion Rate as a percentage
        """
        result = conversion_rate(conversions, total_visitors)
        if isinstance(result, (int, float)):
            self._state['conversion_rate'] = result
        return result
    
    def customer_engagement_score(
        self,
        active_days: Union[int, float],
        total_days: Union[int, float]
    ) -> float:
        """
        Calculate Customer Engagement Score.
        
        Customer Engagement Score measures how actively customers are using a product or service.
        
        Parameters
        ----------
        active_days : int or float
            Number of days the customer was active
        total_days : int or float
            Total number of days in the period
            
        Returns
        -------
        float
            Customer Engagement Score as a percentage
        """
        result = customer_engagement_score(active_days, total_days)
        self._state['customer_engagement_score'] = result
        return result
    
    def daily_active_users_ratio(
        self,
        daily_active_users: Union[int, float],
        total_users: Union[int, float]
    ) -> float:
        """
        Calculate Daily Active Users (DAU) Ratio.
        
        DAU Ratio measures the percentage of total users who are active on a daily basis.
        
        Parameters
        ----------
        daily_active_users : int or float
            Number of daily active users
        total_users : int or float
            Total number of users
            
        Returns
        -------
        float
            Daily Active Users Ratio as a percentage
        """
        result = daily_active_users_ratio(daily_active_users, total_users)
        self._state['daily_active_users_ratio'] = result
        return result
    
    def monthly_active_users_ratio(
        self,
        monthly_active_users: Union[int, float],
        total_users: Union[int, float]
    ) -> float:
        """
        Calculate Monthly Active Users (MAU) Ratio.
        
        MAU Ratio measures the percentage of total users who are active on a monthly basis.
        
        Parameters
        ----------
        monthly_active_users : int or float
            Number of monthly active users
        total_users : int or float
            Total number of users
            
        Returns
        -------
        float
            Monthly Active Users Ratio as a percentage
        """
        result = monthly_active_users_ratio(monthly_active_users, total_users)
        self._state['monthly_active_users_ratio'] = result
        return result
    
    def stickiness_ratio(
        self,
        daily_active_users: Union[int, float],
        monthly_active_users: Union[int, float]
    ) -> float:
        """
        Calculate Stickiness Ratio (DAU/MAU).
        
        Stickiness Ratio measures how frequently active users engage with a product.
        
        Parameters
        ----------
        daily_active_users : int or float
            Number of daily active users
        monthly_active_users : int or float
            Number of monthly active users
            
        Returns
        -------
        float
            Stickiness Ratio as a percentage
        """
        result = stickiness_ratio(daily_active_users, monthly_active_users)
        self._state['stickiness_ratio'] = result
        return result
    
    def gross_margin(
        self,
        revenue: Union[int, float],
        cost_of_goods_sold: Union[int, float]
    ) -> float:
        """
        Calculate Gross Margin.
        
        Gross Margin is the percentage of revenue that exceeds the cost of goods sold.
        
        Parameters
        ----------
        revenue : int or float
            Total revenue
        cost_of_goods_sold : int or float
            Cost of goods sold
            
        Returns
        -------
        float
            Gross Margin as a percentage
        """
        result = gross_margin(revenue, cost_of_goods_sold)
        self._state['gross_margin'] = result
        return result
    
    def burn_rate(
        self,
        starting_capital: Union[int, float],
        ending_capital: Union[int, float],
        months: Union[int, float]
    ) -> float:
        """
        Calculate Monthly Burn Rate.
        
        Burn Rate is the rate at which a company is losing money.
        
        Parameters
        ----------
        starting_capital : int or float
            Capital at the start of the period
        ending_capital : int or float
            Capital at the end of the period
        months : int or float
            Number of months in the period
            
        Returns
        -------
        float
            Monthly Burn Rate
        """
        result = burn_rate(starting_capital, ending_capital, months)
        self._state['burn_rate'] = result
        return result
    
    def runway(
        self,
        current_capital: Union[int, float],
        monthly_burn_rate: Union[int, float]
    ) -> float:
        """
        Calculate Runway in months.
        
        Runway is the amount of time a company has before it runs out of money.
        
        Parameters
        ----------
        current_capital : int or float
            Current capital
        monthly_burn_rate : int or float
            Monthly burn rate
            
        Returns
        -------
        float
            Runway in months
        """
        result = runway(current_capital, monthly_burn_rate)
        self._state['runway'] = result
        return result
    
    def virality_coefficient(
        self,
        new_users: Union[int, float],
        invites_sent: Union[int, float],
        total_users: Union[int, float]
    ) -> float:
        """
        Calculate Virality Coefficient (K-factor).
        
        Virality Coefficient measures how many new users each existing user brings in.
        
        Parameters
        ----------
        new_users : int or float
            Number of new users from invites
        invites_sent : int or float
            Number of invites sent
        total_users : int or float
            Total number of users
            
        Returns
        -------
        float
            Virality Coefficient
        """
        result = virality_coefficient(new_users, invites_sent, total_users)
        self._state['virality_coefficient'] = result
        return result
    
    def time_to_value(
        self,
        onboarding_time: Union[int, float],
        setup_time: Union[int, float],
        learning_time: Union[int, float]
    ) -> float:
        """
        Calculate Time to Value (TTV).
        
        Time to Value is the amount of time it takes for a customer to realize value from a product.
        
        Parameters
        ----------
        onboarding_time : int or float
            Time spent on onboarding
        setup_time : int or float
            Time spent on setup
        learning_time : int or float
            Time spent on learning
            
        Returns
        -------
        float
            Time to Value
        """
        result = time_to_value(onboarding_time, setup_time, learning_time)
        self._state['time_to_value'] = result
        return result
    
    def feature_adoption_rate(
        self,
        users_adopting_feature: Union[int, float],
        total_users: Union[int, float]
    ) -> float:
        """
        Calculate Feature Adoption Rate.
        
        Feature Adoption Rate measures the percentage of users who adopt a specific feature.
        
        Parameters
        ----------
        users_adopting_feature : int or float
            Number of users who adopted the feature
        total_users : int or float
            Total number of users
            
        Returns
        -------
        float
            Feature Adoption Rate as a percentage
        """
        result = feature_adoption_rate(users_adopting_feature, total_users)
        self._state['feature_adoption_rate'] = result
        return result
    
    def roi(
        self,
        revenue: Union[int, float, np.ndarray, list],
        costs: Union[int, float, np.ndarray, list]
    ) -> Union[float, np.ndarray]:
        """
        Calculate Return on Investment (ROI).
        
        ROI measures the return on an investment relative to its cost.
        
        Parameters
        ----------
        revenue : int, float, array-like
            Revenue or return from the investment
        costs : int, float, array-like
            Cost of the investment
            
        Returns
        -------
        float or numpy.ndarray
            Return on Investment as a percentage
        """
        result = roi(revenue, costs)
        if isinstance(result, (int, float)):
            self._state['roi'] = result
        return result 