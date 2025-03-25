"""
KPI Module

This module provides a class for calculating various business metrics
commonly used in SaaS and subscription-based businesses.
"""

import numpy as np
from typing import Union, Dict
from numpy.typing import ArrayLike, NDArray
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
    def health(self) -> Dict[str, Union[float, str, Dict[str, Dict[str, Union[float, str]]], None]]:
        """
        Calculate and return the overall health of the business based on stored KPIs.
        
        Returns
        -------
        dict
            Dictionary containing health score and individual component scores
        """
        health_score = 0.0
        components = {}
        total_weight = 0.0 
        weight_per_metric = 0.10 
        
        if self._state['churn_rate'] is not None:
            churn_score = max(0, min(100, 100 - self._state['churn_rate']))
            health_score += churn_score * weight_per_metric
            total_weight += weight_per_metric
            components['churn_rate'] = {
                'score': churn_score,
                'status': 'Excellent' if churn_score >= 95 else 'Good' if churn_score >= 85 else 'Fair' if churn_score >= 70 else 'Poor' if churn_score >= 50 else 'Critical'
            }
            
        if self._state['retention_rate'] is not None:
            retention_score = self._state['retention_rate']
            health_score += retention_score * weight_per_metric
            total_weight += weight_per_metric
            components['retention_rate'] = {
                'score': retention_score,
                'status': 'Excellent' if retention_score >= 95 else 'Good' if retention_score >= 85 else 'Fair' if retention_score >= 70 else 'Poor' if retention_score >= 50 else 'Critical'
            }

        if self._state['ltv_cac_ratio'] is not None:
            ltv_cac_score = min(100, self._state['ltv_cac_ratio'] * 20)
            health_score += ltv_cac_score * weight_per_metric
            total_weight += weight_per_metric
            components['ltv_cac_ratio'] = {
                'score': ltv_cac_score,
                'status': 'Excellent' if ltv_cac_score >= 80 else 'Good' if ltv_cac_score >= 60 else 'Fair' if ltv_cac_score >= 40 else 'Poor' if ltv_cac_score >= 20 else 'Critical'
            }

        if self._state['gross_margin'] is not None:
            margin_score = self._state['gross_margin']
            health_score += margin_score * weight_per_metric
            total_weight += weight_per_metric
            components['gross_margin'] = {
                'score': margin_score,
                'status': 'Excellent' if margin_score >= 80 else 'Good' if margin_score >= 70 else 'Fair' if margin_score >= 50 else 'Poor' if margin_score >= 30 else 'Critical'
            }

        if self._state['net_promoter_score'] is not None:
            nps_score = max(0, min(100, self._state['net_promoter_score'] + 50))
            health_score += nps_score * weight_per_metric
            total_weight += weight_per_metric
            components['net_promoter_score'] = {
                'score': nps_score,
                'status': 'Excellent' if nps_score >= 60 else 'Good' if nps_score >= 50 else 'Fair' if nps_score >= 30 else 'Poor' if nps_score >= 20 else 'Critical'
            }

        if self._state['customer_satisfaction_score'] is not None:
            csat_score = self._state['customer_satisfaction_score']
            health_score += csat_score * weight_per_metric
            total_weight += weight_per_metric
            components['customer_satisfaction_score'] = {
                'score': csat_score,
                'status': 'Excellent' if csat_score >= 90 else 'Good' if csat_score >= 85 else 'Fair' if csat_score >= 70 else 'Poor' if csat_score >= 50 else 'Critical'
            }

        if self._state['expansion_revenue_rate'] is not None:
            expansion_score = min(100, max(0, self._state['expansion_revenue_rate'] * 2))
            health_score += expansion_score * weight_per_metric
            total_weight += weight_per_metric
            components['expansion_revenue_rate'] = {
                'score': expansion_score,
                'status': 'Excellent' if expansion_score >= 30 else 'Good' if expansion_score >= 20 else 'Fair' if expansion_score >= 10 else 'Poor' if expansion_score >= 5 else 'Critical'
            }

        if self._state['stickiness_ratio'] is not None:
            stickiness_score = self._state['stickiness_ratio']
            health_score += stickiness_score * weight_per_metric
            total_weight += weight_per_metric
            components['stickiness_ratio'] = {
                'score': stickiness_score,
                'status': 'Excellent' if stickiness_score >= 70 else 'Good' if stickiness_score >= 50 else 'Fair' if stickiness_score >= 30 else 'Poor' if stickiness_score >= 20 else 'Critical'
            }

        if self._state['runway'] is not None:
            runway_score = min(100, max(0, self._state['runway'] * 5))
            health_score += runway_score * weight_per_metric
            total_weight += weight_per_metric
            components['runway'] = {
                'score': runway_score,
                'status': 'Excellent' if runway_score >= 18 else 'Good' if runway_score >= 12 else 'Fair' if runway_score >= 6 else 'Poor' if runway_score >= 3 else 'Critical'
            }

        if self._state['roi'] is not None:
            roi_score = min(100, max(0, self._state['roi']))
            health_score += roi_score * weight_per_metric
            total_weight += weight_per_metric
            components['roi'] = {
                'score': roi_score,
                'status': 'Excellent' if roi_score >= 50 else 'Good' if roi_score >= 30 else 'Fair' if roi_score >= 15 else 'Poor' if roi_score >= 5 else 'Critical'
            }

        final_score = (health_score / total_weight) if total_weight > 0 else None

        return {
            'overall_score': final_score,
            'status': ('Excellent' if final_score >= 90 
                      else 'Good' if final_score >= 75 
                      else 'Fair' if final_score >= 60 
                      else 'Poor' if final_score >= 45 
                      else 'Critical') if final_score is not None else 'Not enough data',
            'components': components,
            'metrics_counted': round(total_weight / weight_per_metric)  
        }
    
    def churn_rate(
        self,
        customers_start: ArrayLike,
        customers_end: ArrayLike,
        new_customers: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate customer churn rate.
        
        Churn rate is the percentage of customers who stop using your product or service
        during a given time period.
        
        Parameters
        ----------
        customers_start : array-like
            Number of customers at the start of the period
        customers_end : array-like
            Number of customers at the end of the period
        new_customers : array-like
            Number of new customers acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Churn rate as a percentage
            
        Examples
        --------
        >>> churn_rate(100, 90, 10)
        20.0
        """
        result = churn_rate(customers_start, customers_end, new_customers)
        if isinstance(result, (int, float)):
            self._state['churn_rate'] = result
        return result
    
    def retention_rate(
        self,
        customers_start: ArrayLike,
        customers_end: ArrayLike,
        new_customers: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate customer retention rate.
        
        Retention rate is the percentage of customers who remain with your product or service
        over a given time period.
        
        Parameters
        ----------
        customers_start : array-like
            Number of customers at the start of the period
        customers_end : array-like
            Number of customers at the end of the period
        new_customers : array-like
            Number of new customers acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Retention rate as a percentage
            
        Examples
        --------
        >>> retention_rate(100, 90, 10)
        80.0
        """
        result = retention_rate(customers_start, customers_end, new_customers)
        if isinstance(result, (int, float)):
            self._state['retention_rate'] = result
        return result
    
    def customer_lifetime_value(
        self,
        avg_revenue_per_customer: ArrayLike,
        gross_margin: ArrayLike,
        churn_rate_value: ArrayLike,
        discount_rate: ArrayLike = 10.0
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Customer Lifetime Value (CLV).
        
        CLV is the total worth to a business of a customer over the whole period of their relationship.
        
        Parameters
        ----------
        avg_revenue_per_customer : array-like
            Average revenue per customer per period (e.g., monthly)
        gross_margin : array-like
            Gross margin percentage (0-100)
        churn_rate_value : array-like
            Churn rate percentage (0-100)
        discount_rate : array-like, default 10.0
            Annual discount rate for future cash flows (0-100)
            
        Returns
        -------
        float or numpy.ndarray
            Customer Lifetime Value
            
        Examples
        --------
        >>> customer_lifetime_value(100, 70, 5, 10)
        466.67
        """
        result = customer_lifetime_value(
            avg_revenue_per_customer, gross_margin, churn_rate_value, discount_rate
        )
        if isinstance(result, (int, float)):
            self._state['customer_lifetime_value'] = result
        return result
    
    def customer_acquisition_cost(
        self,
        marketing_costs: ArrayLike,
        sales_costs: ArrayLike,
        new_customers: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Customer Acquisition Cost (CAC).
        
        CAC is the cost of convincing a potential customer to buy a product or service.
        
        Parameters
        ----------
        marketing_costs : array-like
            Total marketing costs for the period
        sales_costs : array-like
            Total sales costs for the period
        new_customers : array-like
            Number of new customers acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Customer Acquisition Cost
            
        Examples
        --------
        >>> customer_acquisition_cost(5000, 3000, 100)
        80.0
        """
        result = customer_acquisition_cost(marketing_costs, sales_costs, new_customers)
        if isinstance(result, (int, float)):
            self._state['customer_acquisition_cost'] = result
        return result
    
    def monthly_recurring_revenue(
        self,
        paying_customers: ArrayLike,
        avg_revenue_per_customer: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Monthly Recurring Revenue (MRR).
        
        MRR is the predictable total revenue generated by all the active subscriptions in a month.
        
        Parameters
        ----------
        paying_customers : array-like
            Number of paying customers
        avg_revenue_per_customer : array-like
            Average revenue per customer per month
            
        Returns
        -------
        float or numpy.ndarray
            Monthly Recurring Revenue
            
        Examples
        --------
        >>> monthly_recurring_revenue(100, 50)
        5000.0
        """
        result = monthly_recurring_revenue(paying_customers, avg_revenue_per_customer)
        if isinstance(result, (int, float)):
            self._state['monthly_recurring_revenue'] = result
        return result
    
    def annual_recurring_revenue(
        self,
        paying_customers: ArrayLike,
        avg_revenue_per_customer: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Annual Recurring Revenue (ARR).
        
        ARR is the value of the recurring revenue of a business's term subscriptions normalized for a single calendar year.
        
        Parameters
        ----------
        paying_customers : array-like
            Number of paying customers
        avg_revenue_per_customer : array-like
            Average revenue per customer per month
            
        Returns
        -------
        float or numpy.ndarray
            Annual Recurring Revenue
            
        Examples
        --------
        >>> annual_recurring_revenue(100, 50)
        60000.0
        """
        result = annual_recurring_revenue(paying_customers, avg_revenue_per_customer)
        if isinstance(result, (int, float)):
            self._state['annual_recurring_revenue'] = result
        return result
    
    def net_promoter_score(
        self,
        promoters: ArrayLike,
        detractors: ArrayLike,
        total_respondents: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Net Promoter Score (NPS).
        
        NPS measures customer experience and predicts business growth.
        
        Parameters
        ----------
        promoters : array-like
            Number of promoters (customers who rated 9-10)
        detractors : array-like
            Number of detractors (customers who rated 0-6)
        total_respondents : array-like
            Total number of survey respondents
            
        Returns
        -------
        float or numpy.ndarray
            Net Promoter Score (ranges from -100 to 100)
            
        Examples
        --------
        >>> net_promoter_score(70, 10, 100)
        60.0
        """
        result = net_promoter_score(promoters, detractors, total_respondents)
        if isinstance(result, (int, float)):
            self._state['net_promoter_score'] = result
        return result
    
    def revenue_churn_rate(
        self,
        revenue_start: ArrayLike,
        revenue_end: ArrayLike,
        new_revenue: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Revenue Churn Rate.
        
        Revenue Churn Rate is the percentage of revenue lost from existing customers in a given period.
        
        Parameters
        ----------
        revenue_start : array-like
            Revenue at the start of the period
        revenue_end : array-like
            Revenue at the end of the period
        new_revenue : array-like
            New revenue acquired during the period
            
        Returns
        -------
        float or numpy.ndarray
            Revenue Churn Rate as a percentage
            
        Examples
        --------
        >>> revenue_churn_rate(10000, 9500, 1000)
        15.0
        """
        result = revenue_churn_rate(revenue_start, revenue_end, new_revenue)
        if isinstance(result, (int, float)):
            self._state['revenue_churn_rate'] = result
        return result
    
    def expansion_revenue_rate(
        self,
        upsell_revenue: ArrayLike,
        cross_sell_revenue: ArrayLike,
        revenue_start: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Expansion Revenue Rate.
        
        Expansion Revenue Rate is the percentage of additional revenue generated from existing customers.
        
        Parameters
        ----------
        upsell_revenue : array-like
            Revenue from upselling to existing customers
        cross_sell_revenue : array-like
            Revenue from cross-selling to existing customers
        revenue_start : array-like
            Revenue at the start of the period
            
        Returns
        -------
        float or numpy.ndarray
            Expansion Revenue Rate as a percentage
            
        Examples
        --------
        >>> expansion_revenue_rate(1000, 500, 10000)
        15.0
        """
        result = expansion_revenue_rate(upsell_revenue, cross_sell_revenue, revenue_start)
        if isinstance(result, (int, float)):
            self._state['expansion_revenue_rate'] = result
        return result
    
    def ltv_cac_ratio(
        self,
        ltv: ArrayLike,
        cac: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate LTV:CAC Ratio.
        
        LTV:CAC Ratio is a metric that compares the lifetime value of a customer to the cost of acquiring that customer.
        
        Parameters
        ----------
        ltv : array-like
            Customer Lifetime Value
        cac : array-like
            Customer Acquisition Cost
            
        Returns
        -------
        float or numpy.ndarray
            LTV:CAC Ratio
            
        Examples
        --------
        >>> ltv_cac_ratio(1000, 200)
        5.0
        """
        result = ltv_cac_ratio(ltv, cac)
        if isinstance(result, (int, float)):
            self._state['ltv_cac_ratio'] = result
        return result
    
    def payback_period(
        self,
        cac: ArrayLike,
        avg_monthly_revenue: ArrayLike,
        gross_margin: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate CAC Payback Period in months.
        
        CAC Payback Period is the number of months it takes to recover the cost of acquiring a customer.
        
        Parameters
        ----------
        cac : array-like
            Customer Acquisition Cost
        avg_monthly_revenue : array-like
            Average monthly revenue per customer
        gross_margin : array-like
            Gross margin percentage (0-100)
            
        Returns
        -------
        float or numpy.ndarray
            CAC Payback Period in months
            
        Examples
        --------
        >>> payback_period(1000, 100, 70)
        14.29
        """
        result = payback_period(cac, avg_monthly_revenue, gross_margin)
        if isinstance(result, (int, float)):
            self._state['payback_period'] = result
        return result
    
    def customer_satisfaction_score(
        self,
        satisfaction_ratings: ArrayLike,
        max_rating: ArrayLike = 5
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Customer Satisfaction Score (CSAT).
        
        CSAT measures how satisfied customers are with a product, service, or interaction.
        
        Parameters
        ----------
        satisfaction_ratings : array-like
            Array of customer satisfaction ratings
        max_rating : array-like, default 5
            Maximum possible rating value
            
        Returns
        -------
        float or numpy.ndarray
            Customer Satisfaction Score as a percentage
            
        Examples
        --------
        >>> customer_satisfaction_score([4, 5, 3, 5, 4])
        84.0
        """
        result = customer_satisfaction_score(satisfaction_ratings, max_rating)
        if isinstance(result, (int, float)):
            self._state['customer_satisfaction_score'] = result
        return result
    
    def customer_effort_score(
        self,
        effort_ratings: ArrayLike,
        max_rating: ArrayLike = 7
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Customer Effort Score (CES).
        
        CES measures how much effort a customer has to exert to use a product or service.
        Lower scores are better.
        
        Parameters
        ----------
        effort_ratings : array-like
            Array of customer effort ratings
        max_rating : array-like, default 7
            Maximum possible rating value
            
        Returns
        -------
        float or numpy.ndarray
            Customer Effort Score (average)
            
        Examples
        --------
        >>> customer_effort_score([2, 3, 1, 2, 4])
        2.4
        """
        result = customer_effort_score(effort_ratings, max_rating)
        if isinstance(result, (int, float)):
            self._state['customer_effort_score'] = result
        return result
    
    def average_revenue_per_user(
        self,
        total_revenue: ArrayLike,
        total_users: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Average Revenue Per User (ARPU).
        
        ARPU measures the average revenue generated per user or customer.
        
        Parameters
        ----------
        total_revenue : array-like
            Total revenue for the period
        total_users : array-like
            Total number of users or customers
            
        Returns
        -------
        float or numpy.ndarray
            Average Revenue Per User
            
        Examples
        --------
        >>> average_revenue_per_user(10000, 500)
        20.0
        """
        result = average_revenue_per_user(total_revenue, total_users)
        if isinstance(result, (int, float)):
            self._state['average_revenue_per_user'] = result
        return result
    
    def average_revenue_per_paying_user(
        self,
        total_revenue: ArrayLike,
        paying_users: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Average Revenue Per Paying User (ARPPU).
        
        ARPPU measures the average revenue generated per paying user or customer.
        
        Parameters
        ----------
        total_revenue : array-like
            Total revenue for the period
        paying_users : array-like
            Number of paying users or customers
            
        Returns
        -------
        float or numpy.ndarray
            Average Revenue Per Paying User
            
        Examples
        --------
        >>> average_revenue_per_paying_user(10000, 200)
        50.0
        """
        result = average_revenue_per_paying_user(total_revenue, paying_users)
        if isinstance(result, (int, float)):
            self._state['average_revenue_per_paying_user'] = result
        return result
    
    def conversion_rate(
        self,
        conversions: ArrayLike,
        total_visitors: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Conversion Rate.
        
        Conversion Rate is the percentage of visitors who take a desired action.
        
        Parameters
        ----------
        conversions : array-like
            Number of conversions (desired actions taken)
        total_visitors : array-like
            Total number of visitors or users
            
        Returns
        -------
        float or numpy.ndarray
            Conversion Rate as a percentage
            
        Examples
        --------
        >>> conversion_rate(50, 1000)
        5.0
        """
        result = conversion_rate(conversions, total_visitors)
        if isinstance(result, (int, float)):
            self._state['conversion_rate'] = result
        return result
    
    def customer_engagement_score(
        self,
        active_days: ArrayLike,
        total_days: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Customer Engagement Score.
        
        Customer Engagement Score measures how actively customers are using a product or service.
        
        Parameters
        ----------
        active_days : array-like
            Number of days the customer was active
        total_days : array-like
            Total number of days in the period
            
        Returns
        -------
        float or numpy.ndarray
            Customer Engagement Score as a percentage
            
        Examples
        --------
        >>> customer_engagement_score(15, 30)
        50.0
        """
        result = customer_engagement_score(active_days, total_days)
        if isinstance(result, (int, float)):
            self._state['customer_engagement_score'] = result
        return result
    
    def daily_active_users_ratio(
        self,
        daily_active_users: ArrayLike,
        total_users: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Daily Active Users (DAU) Ratio.
        
        DAU Ratio measures the percentage of total users who are active on a daily basis.
        
        Parameters
        ----------
        daily_active_users : array-like
            Number of daily active users
        total_users : array-like
            Total number of users
            
        Returns
        -------
        float or numpy.ndarray
            Daily Active Users Ratio as a percentage
            
        Examples
        --------
        >>> daily_active_users_ratio(500, 2000)
        25.0
        """
        result = daily_active_users_ratio(daily_active_users, total_users)
        if isinstance(result, (int, float)):
            self._state['daily_active_users_ratio'] = result
        return result
    
    def monthly_active_users_ratio(
        self,
        monthly_active_users: ArrayLike,
        total_users: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Monthly Active Users (MAU) Ratio.
        
        MAU Ratio measures the percentage of total users who are active on a monthly basis.
        
        Parameters
        ----------
        monthly_active_users : array-like
            Number of monthly active users
        total_users : array-like
            Total number of users
            
        Returns
        -------
        float or numpy.ndarray
            Monthly Active Users Ratio as a percentage
            
        Examples
        --------
        >>> monthly_active_users_ratio(1500, 2000)
        75.0
        """
        result = monthly_active_users_ratio(monthly_active_users, total_users)
        if isinstance(result, (int, float)):
            self._state['monthly_active_users_ratio'] = result
        return result
    
    def stickiness_ratio(
        self,
        daily_active_users: ArrayLike,
        monthly_active_users: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Stickiness Ratio (DAU/MAU).
        
        Stickiness Ratio measures how frequently active users engage with a product.
        
        Parameters
        ----------
        daily_active_users : array-like
            Number of daily active users
        monthly_active_users : array-like
            Number of monthly active users
            
        Returns
        -------
        float or numpy.ndarray
            Stickiness Ratio as a percentage
            
        Examples
        --------
        >>> stickiness_ratio(500, 1500)
        33.33
        """
        result = stickiness_ratio(daily_active_users, monthly_active_users)
        if isinstance(result, (int, float)):
            self._state['stickiness_ratio'] = result
        return result
    
    def gross_margin(
        self,
        revenue: ArrayLike,
        cost_of_goods_sold: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Gross Margin.
        
        Gross Margin is the percentage of revenue that exceeds the cost of goods sold.
        
        Parameters
        ----------
        revenue : array-like
            Total revenue
        cost_of_goods_sold : array-like
            Cost of goods sold
            
        Returns
        -------
        float or numpy.ndarray
            Gross Margin as a percentage
            
        Examples
        --------
        >>> gross_margin(10000, 3000)
        70.0
        """
        result = gross_margin(revenue, cost_of_goods_sold)
        if isinstance(result, (int, float)):
            self._state['gross_margin'] = result
        return result
    
    def burn_rate(
        self,
        starting_capital: ArrayLike,
        ending_capital: ArrayLike,
        months: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Monthly Burn Rate.
        
        Burn Rate is the rate at which a company is losing money.
        
        Parameters
        ----------
        starting_capital : array-like
            Capital at the start of the period
        ending_capital : array-like
            Capital at the end of the period
        months : array-like
            Number of months in the period
            
        Returns
        -------
        float or numpy.ndarray
            Monthly Burn Rate
            
        Examples
        --------
        >>> burn_rate(100000, 70000, 6)
        5000.0
        """
        result = burn_rate(starting_capital, ending_capital, months)
        if isinstance(result, (int, float)):
            self._state['burn_rate'] = result
        return result
    
    def runway(
        self,
        current_capital: ArrayLike,
        monthly_burn_rate: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Runway in months.
        
        Runway is the amount of time a company has before it runs out of money.
        
        Parameters
        ----------
        current_capital : array-like
            Current capital
        monthly_burn_rate : array-like
            Monthly burn rate
            
        Returns
        -------
        float or numpy.ndarray
            Runway in months
            
        Examples
        --------
        >>> runway(100000, 5000)
        20.0
        """
        result = runway(current_capital, monthly_burn_rate)
        if isinstance(result, (int, float)):
            self._state['runway'] = result
        return result
    
    def virality_coefficient(
        self,
        new_users: ArrayLike,
        invites_sent: ArrayLike,
        total_users: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Virality Coefficient (K-factor).
        
        Virality Coefficient measures how many new users each existing user brings in.
        
        Parameters
        ----------
        new_users : array-like
            Number of new users from invites
        invites_sent : array-like
            Number of invites sent
        total_users : array-like
            Total number of users
            
        Returns
        -------
        float or numpy.ndarray
            Virality Coefficient
            
        Examples
        --------
        >>> virality_coefficient(100, 500, 1000)
        0.1
        """
        result = virality_coefficient(new_users, invites_sent, total_users)
        if isinstance(result, (int, float)):
            self._state['virality_coefficient'] = result
        return result
    
    def time_to_value(
        self,
        onboarding_time: ArrayLike,
        setup_time: ArrayLike,
        learning_time: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Time to Value (TTV).
        
        Time to Value is the amount of time it takes for a customer to realize value from a product.
        
        Parameters
        ----------
        onboarding_time : array-like
            Time spent on onboarding
        setup_time : array-like
            Time spent on setup
        learning_time : array-like
            Time spent on learning
            
        Returns
        -------
        float or numpy.ndarray
            Time to Value
            
        Examples
        --------
        >>> time_to_value(2, 3, 5)
        10.0
        """
        result = time_to_value(onboarding_time, setup_time, learning_time)
        if isinstance(result, (int, float)):
            self._state['time_to_value'] = result
        return result
    
    def feature_adoption_rate(
        self,
        users_adopting_feature: ArrayLike,
        total_users: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Feature Adoption Rate.
        
        Feature Adoption Rate measures the percentage of users who adopt a specific feature.
        
        Parameters
        ----------
        users_adopting_feature : array-like
            Number of users who adopted the feature
        total_users : array-like
            Total number of users
            
        Returns
        -------
        float or numpy.ndarray
            Feature Adoption Rate as a percentage
            
        Examples
        --------
        >>> feature_adoption_rate(300, 1000)
        30.0
        """
        result = feature_adoption_rate(users_adopting_feature, total_users)
        if isinstance(result, (int, float)):
            self._state['feature_adoption_rate'] = result
        return result
    
    def roi(
        self,
        revenue: ArrayLike,
        costs: ArrayLike
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate Return on Investment (ROI).
        
        ROI measures the return on an investment relative to its cost.
        
        Parameters
        ----------
        revenue : array-like
            Revenue or return from the investment
        costs : array-like
            Cost of the investment
            
        Returns
        -------
        float or numpy.ndarray
            Return on Investment as a percentage
            
        Examples
        --------
        >>> roi(150, 100)
        50.0
        >>> roi([150, 200, 250], [100, 120, 150])
        array([50., 66.67, 66.67])
        """
        result = roi(revenue, costs)
        if isinstance(result, (int, float)):
            self._state['roi'] = result
        return result 