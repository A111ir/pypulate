"""
Business KPIs Module

This module provides functions for calculating various business metrics
commonly used in SaaS and subscription-based businesses.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict
from ..dtypes.parray import Parray

def churn_rate(
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
        
    Examples
    --------
    >>> churn_rate(100, 90, 10)
    20.0
    """
    if isinstance(customers_start, list):
        customers_start = np.array(customers_start)
    if isinstance(customers_end, list):
        customers_end = np.array(customers_end)
    if isinstance(new_customers, list):
        new_customers = np.array(new_customers)
    
    lost_customers = customers_start + new_customers - customers_end
    
    if np.isscalar(customers_start):
        if customers_start == 0:
            return 0.0
        return (lost_customers / customers_start) * 100.0
    
    result = np.zeros_like(customers_start, dtype=float)
    non_zero_mask = customers_start != 0
    result[non_zero_mask] = (lost_customers[non_zero_mask] / customers_start[non_zero_mask]) * 100.0
    
    return result

def retention_rate(
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
        
    Examples
    --------
    >>> retention_rate(100, 90, 10)
    80.0
    """
    if isinstance(customers_start, list):
        customers_start = np.array(customers_start)
    if isinstance(customers_end, list):
        customers_end = np.array(customers_end)
    if isinstance(new_customers, list):
        new_customers = np.array(new_customers)
    
    churn = churn_rate(customers_start, customers_end, new_customers)
    
    if np.isscalar(churn):
        return 100.0 - churn
    else:
        return 100.0 - churn

def customer_lifetime_value(
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
        
    Examples
    --------
    >>> customer_lifetime_value(100, 70, 5, 10)
    466.66
    """
    gross_margin_decimal = gross_margin / 100.0
    churn_rate_decimal = churn_rate_value / 100.0
    discount_rate_decimal = discount_rate / 100.0
    
    avg_customer_lifespan = 1 / churn_rate_decimal if churn_rate_decimal > 0 else float('inf')
    
    clv = (avg_revenue_per_customer * gross_margin_decimal) / (churn_rate_decimal + discount_rate_decimal)
    
    if churn_rate_decimal == 0:
        max_periods = 240
        clv = 0
        for i in range(max_periods):
            clv += (avg_revenue_per_customer * gross_margin_decimal) / ((1 + discount_rate_decimal/12) ** i)
    
    return clv

def customer_acquisition_cost(
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
        
    Examples
    --------
    >>> customer_acquisition_cost(5000, 3000, 100)
    80.0
    """
    if isinstance(marketing_costs, list):
        marketing_costs = np.array(marketing_costs)
    if isinstance(sales_costs, list):
        sales_costs = np.array(sales_costs)
    if isinstance(new_customers, list):
        new_customers = np.array(new_customers)
    
    total_costs = marketing_costs + sales_costs
    
    if np.isscalar(new_customers):
        if new_customers == 0:
            return float('inf')
        return total_costs / new_customers
    
    result = np.full_like(new_customers, float('inf'), dtype=float)
    non_zero_mask = new_customers != 0
    result[non_zero_mask] = total_costs[non_zero_mask] / new_customers[non_zero_mask]
    
    return result

def monthly_recurring_revenue(
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
        
    Examples
    --------
    >>> monthly_recurring_revenue(100, 50)
    5000.0
    """
    if isinstance(paying_customers, list):
        paying_customers = np.array(paying_customers)
    if isinstance(avg_revenue_per_customer, list):
        avg_revenue_per_customer = np.array(avg_revenue_per_customer)
    
    return paying_customers * avg_revenue_per_customer

def annual_recurring_revenue(
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
        
    Examples
    --------
    >>> annual_recurring_revenue(100, 50)
    60000.0
    """
    return monthly_recurring_revenue(paying_customers, avg_revenue_per_customer) * 12

def net_promoter_score(
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
        
    Examples
    --------
    >>> net_promoter_score(70, 10, 100)
    60.0
    """
    if isinstance(promoters, list):
        promoters = np.array(promoters)
    if isinstance(detractors, list):
        detractors = np.array(detractors)
    if isinstance(total_respondents, list):
        total_respondents = np.array(total_respondents)
    
    if np.isscalar(total_respondents):
        if total_respondents == 0:
            return 0.0
        
        promoters_percent = (promoters / total_respondents) * 100
        detractors_percent = (detractors / total_respondents) * 100
        
        return promoters_percent - detractors_percent
    
    result = np.zeros_like(total_respondents, dtype=float)
    non_zero_mask = total_respondents != 0
    
    promoters_percent = np.zeros_like(total_respondents, dtype=float)
    detractors_percent = np.zeros_like(total_respondents, dtype=float)
    
    promoters_percent[non_zero_mask] = (promoters[non_zero_mask] / total_respondents[non_zero_mask]) * 100
    detractors_percent[non_zero_mask] = (detractors[non_zero_mask] / total_respondents[non_zero_mask]) * 100
    
    result = promoters_percent - detractors_percent
    
    return result

def revenue_churn_rate(
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
        
    Examples
    --------
    >>> revenue_churn_rate(10000, 9500, 1000)
    15.0
    """
    if isinstance(revenue_start, list):
        revenue_start = np.array(revenue_start)
    if isinstance(revenue_end, list):
        revenue_end = np.array(revenue_end)
    if isinstance(new_revenue, list):
        new_revenue = np.array(new_revenue)
    
    lost_revenue = revenue_start + new_revenue - revenue_end
    
    if np.isscalar(revenue_start):
        if revenue_start == 0:
            return 0.0
        return (lost_revenue / revenue_start) * 100.0
    
    result = np.zeros_like(revenue_start, dtype=float)
    non_zero_mask = revenue_start != 0
    result[non_zero_mask] = (lost_revenue[non_zero_mask] / revenue_start[non_zero_mask]) * 100.0
    
    return result

def expansion_revenue_rate(
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
        
    Examples
    --------
    >>> expansion_revenue_rate(1000, 500, 10000)
    15.0
    """
    expansion_revenue = upsell_revenue + cross_sell_revenue
    
    if revenue_start == 0:
        return 0.0
    
    return (expansion_revenue / revenue_start) * 100.0

def ltv_cac_ratio(
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
        
    Examples
    --------
    >>> ltv_cac_ratio(1000, 200)
    5.0
    """
    if cac == 0:
        return float('inf')
    
    return ltv / cac

def payback_period(
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
        
    Examples
    --------
    >>> payback_period(1000, 100, 70)
    14.29
    """
    gross_margin_decimal = gross_margin / 100.0
    
    monthly_gross_profit = avg_monthly_revenue * gross_margin_decimal
    
    if monthly_gross_profit == 0:
        return float('inf')
    
    return cac / monthly_gross_profit


def customer_satisfaction_score(
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
        
    Examples
    --------
    >>> customer_satisfaction_score([4, 5, 3, 5, 4])
    84.0
    """

    if isinstance(satisfaction_ratings, list):
        ratings = np.array(satisfaction_ratings)
    else:
        ratings = satisfaction_ratings
    
    if len(ratings) == 0:
        return 0.0
    
    avg_rating = np.mean(ratings)
    
    return (avg_rating / max_rating) * 100.0

def customer_effort_score(
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
        
    Examples
    --------
    >>> customer_effort_score([2, 3, 1, 2, 4])
    2.4
    """
    if isinstance(effort_ratings, list):
        ratings = np.array(effort_ratings)
    else:
        ratings = effort_ratings
    
    if len(ratings) == 0:
        return 0.0
    
    return np.mean(ratings)

def average_revenue_per_user(
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
        
    Examples
    --------
    >>> average_revenue_per_user(10000, 500)
    20.0
    """
    if isinstance(total_revenue, list):
        total_revenue = np.array(total_revenue)
    if isinstance(total_users, list):
        total_users = np.array(total_users)
    
    if np.isscalar(total_users):
        if total_users == 0:
            return 0.0
        return total_revenue / total_users
    
    result = np.zeros_like(total_users, dtype=float)
    non_zero_mask = total_users != 0
    result[non_zero_mask] = total_revenue[non_zero_mask] / total_users[non_zero_mask]
    
    return result

def average_revenue_per_paying_user(
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
        
    Examples
    --------
    >>> average_revenue_per_paying_user(10000, 200)
    50.0
    """
    if paying_users == 0:
        return 0.0
    
    return total_revenue / paying_users

def conversion_rate(
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
        
    Examples
    --------
    >>> conversion_rate(50, 1000)
    5.0
    """
    if isinstance(conversions, list):
        conversions = np.array(conversions)
    if isinstance(total_visitors, list):
        total_visitors = np.array(total_visitors)
    
    if np.isscalar(total_visitors):
        if total_visitors == 0:
            return 0.0
        return (conversions / total_visitors) * 100.0
    
    result = np.zeros_like(total_visitors, dtype=float)
    non_zero_mask = total_visitors != 0
    result[non_zero_mask] = (conversions[non_zero_mask] / total_visitors[non_zero_mask]) * 100.0
    
    return result

def customer_engagement_score(
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
        
    Examples
    --------
    >>> customer_engagement_score(15, 30)
    50.0
    """
    if total_days == 0:
        return 0.0
    
    return (active_days / total_days) * 100.0

def daily_active_users_ratio(
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
        
    Examples
    --------
    >>> daily_active_users_ratio(500, 2000)
    25.0
    """
    if total_users == 0:
        return 0.0
    
    return (daily_active_users / total_users) * 100.0

def monthly_active_users_ratio(
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
        
    Examples
    --------
    >>> monthly_active_users_ratio(1500, 2000)
    75.0
    """
    if total_users == 0:
        return 0.0
    
    return (monthly_active_users / total_users) * 100.0

def stickiness_ratio(
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
        
    Examples
    --------
    >>> stickiness_ratio(500, 1500)
    33.33
    """
    if monthly_active_users == 0:
        return 0.0
    
    return (daily_active_users / monthly_active_users) * 100.0

def gross_margin(
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
        
    Examples
    --------
    >>> gross_margin(10000, 3000)
    70.0
    """
    if revenue == 0:
        return 0.0
    
    gross_profit = revenue - cost_of_goods_sold
    return (gross_profit / revenue) * 100.0

def burn_rate(
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
        
    Examples
    --------
    >>> burn_rate(100000, 70000, 6)
    5000.0
    """
    if months == 0:
        return 0.0
    
    capital_used = starting_capital - ending_capital
    return capital_used / months

def runway(
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
        
    Examples
    --------
    >>> runway(100000, 5000)
    20.0
    """
    if monthly_burn_rate == 0:
        return float('inf')
    
    return current_capital / monthly_burn_rate

def virality_coefficient(
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
        
    Examples
    --------
    >>> virality_coefficient(100, 500, 1000)
    0.1
    """
    if total_users == 0 or invites_sent == 0:
        return 0.0
    
    invites_per_user = invites_sent / total_users
    conversion_rate_val = new_users / invites_sent
    
    return invites_per_user * conversion_rate_val

def time_to_value(
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
        
    Examples
    --------
    >>> time_to_value(2, 3, 5)
    10.0
    """
    return onboarding_time + setup_time + learning_time

def feature_adoption_rate(
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
        
    Examples
    --------
    >>> feature_adoption_rate(300, 1000)
    30.0
    """
    if total_users == 0:
        return 0.0
    
    return (users_adopting_feature / total_users) * 100.0

def roi(
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
        
    Examples
    --------
    >>> roi(150, 100)
    50.0
    >>> roi([150, 200, 250], [100, 120, 150])
    array([50., 66.67, 66.67])
    """
    if isinstance(revenue, list):
        revenue = np.array(revenue)
    if isinstance(costs, list):
        costs = np.array(costs)
    
    if np.isscalar(revenue) and np.isscalar(costs):
        if costs == 0:
            return 0.0
        return ((revenue - costs) / costs) * 100.0
    
    result = np.zeros_like(costs, dtype=float)
    non_zero_mask = costs != 0
    result[non_zero_mask] = ((revenue[non_zero_mask] - costs[non_zero_mask]) / costs[non_zero_mask]) * 100.0
    
    return result 