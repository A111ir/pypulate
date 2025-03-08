"""
Credit Scoring Class for Pypulate

This module provides a class for credit scoring models and risk assessment tools.
"""

from pypulate.credit import altman_z_score, merton_model, debt_service_coverage_ratio
from pypulate.credit import weight_of_evidence, logistic_regression_score, create_scorecard
from pypulate.credit import transition_matrix, financial_ratios
from pypulate.credit import expected_credit_loss, loan_pricing, scoring_model_validation
from pypulate.credit import loss_given_default, exposure_at_default


class CreditScoring:
    """
    Credit scoring models for risk assessment and creditworthiness evaluation.
    Implements various statistical and financial models for credit risk analysis.
    """

    def __init__(self):
        """Initialize the CreditScoring class."""
        self._history = []

    def altman_z_score(self, working_capital, retained_earnings, ebit, 
                       market_value_equity, sales, total_assets, total_liabilities):
        """
        Calculate Altman Z-Score for predicting bankruptcy risk.
        
        Z-Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
        
        Parameters
        ----------
        working_capital : float
            Working capital
        retained_earnings : float
            Retained earnings
        ebit : float
            Earnings before interest and taxes
        market_value_equity : float
            Market value of equity
        sales : float
            Sales
        total_assets : float
            Total assets
        total_liabilities : float
            Total liabilities
            
        Returns
        -------
        dict
            Z-score value and risk interpretation
        """
        result = altman_z_score(
            working_capital, retained_earnings, ebit, 
            market_value_equity, sales, total_assets, total_liabilities
        )
        self._history.append({"model": "altman_z_score", "result": result})
        return result
    
    def merton_model(self, asset_value, debt_face_value, asset_volatility, 
                     risk_free_rate, time_to_maturity):
        """
        Calculate default probability using the Merton model.
        
        Parameters
        ----------
        asset_value : float
            Market value of assets
        debt_face_value : float
            Face value of debt
        asset_volatility : float
            Volatility of assets (annualized)
        risk_free_rate : float
            Risk-free interest rate
        time_to_maturity : float
            Time to maturity in years
            
        Returns
        -------
        dict
            Default probability and distance to default
        """
        result = merton_model(
            asset_value, debt_face_value, asset_volatility, 
            risk_free_rate, time_to_maturity
        )
        self._history.append({"model": "merton_model", "result": result})
        return result
    
    def debt_service_coverage_ratio(self, net_operating_income, total_debt_service):
        """
        Calculate Debt Service Coverage Ratio (DSCR).
        
        DSCR = Net Operating Income / Total Debt Service
        
        Parameters
        ----------
        net_operating_income : float
            Net operating income
        total_debt_service : float
            Total debt service
            
        Returns
        -------
        dict
            DSCR value and interpretation
        """
        result = debt_service_coverage_ratio(net_operating_income, total_debt_service)
        self._history.append({"model": "debt_service_coverage_ratio", "result": result})
        return result
    
    def weight_of_evidence(self, good_count, bad_count, min_samples=0.01, adjustment=0.5):
        """
        Calculate Weight of Evidence (WOE) and Information Value (IV).
        
        WOE = ln(Distribution of Good / Distribution of Bad)
        
        Parameters
        ----------
        good_count : array_like
            Count of good cases in each bin
        bad_count : array_like
            Count of bad cases in each bin
        min_samples : float, optional
            Minimum percentage of samples required in a bin
        adjustment : float, optional
            Adjustment factor for zero counts
            
        Returns
        -------
        dict
            WOE values, IV, and distributions
        """
        result = weight_of_evidence(good_count, bad_count, min_samples, adjustment)
        self._history.append({"model": "weight_of_evidence", "result": result})
        return result
    
    def logistic_regression_score(self, coefficients, features, intercept=0):
        """
        Calculate credit score using logistic regression.
        
        Parameters
        ----------
        coefficients : array_like
            Logistic regression coefficients
        features : array_like
            Feature values
        intercept : float, optional
            Intercept term
            
        Returns
        -------
        dict
            Credit score and probability of default
        """
        result = logistic_regression_score(coefficients, features, intercept)
        self._history.append({"model": "logistic_regression_score", "result": result})
        return result
    
    def create_scorecard(self, features, weights, offsets=None, scaling_factor=100.0, base_score=600):
        """
        Create a points-based scorecard.
        
        Parameters
        ----------
        features : dict
            Dictionary of feature names and values
        weights : dict
            Dictionary of feature names and weights
        offsets : dict, optional
            Dictionary of feature names and offsets
        scaling_factor : float, optional
            Scaling factor for points
        base_score : float, optional
            Base score
            
        Returns
        -------
        dict
            Total score and points breakdown
        """
        result = create_scorecard(features, weights, offsets, scaling_factor, base_score)
        self._history.append({"model": "create_scorecard", "result": result})
        return result
    
    def transition_matrix(self, ratings_t0, ratings_t1):
        """
        Calculate credit rating transition matrix.
        
        Parameters
        ----------
        ratings_t0 : array_like
            Ratings at time 0
        ratings_t1 : array_like
            Ratings at time 1
            
        Returns
        -------
        dict
            Transition matrix and probabilities
        """
        result = transition_matrix(ratings_t0, ratings_t1)
        self._history.append({"model": "transition_matrix", "result": result})
        return result
    
    def financial_ratios(self, current_assets, current_liabilities, total_assets, 
                        total_liabilities, ebit, interest_expense, net_income, 
                        total_equity, sales):
        """
        Calculate key financial ratios for credit assessment.
        
        Parameters
        ----------
        current_assets : float
            Current assets
        current_liabilities : float
            Current liabilities
        total_assets : float
            Total assets
        total_liabilities : float
            Total liabilities
        ebit : float
            Earnings before interest and taxes
        interest_expense : float
            Interest expense
        net_income : float
            Net income
        total_equity : float
            Total equity
        sales : float
            Sales
            
        Returns
        -------
        dict
            Financial ratios and assessments
        """
        result = financial_ratios(
            current_assets, current_liabilities, total_assets, 
            total_liabilities, ebit, interest_expense, net_income, 
            total_equity, sales
        )
        self._history.append({"model": "financial_ratios", "result": result})
        return result
    
    def expected_credit_loss(self, pd, lgd, ead, time_horizon=1.0, discount_rate=0.0):
        """
        Calculate expected credit loss.
        
        ECL = PD × LGD × EAD × Discount Factor
        
        Parameters
        ----------
        pd : float
            Probability of default
        lgd : float
            Loss given default (as a decimal)
        ead : float
            Exposure at default
        time_horizon : float, optional
            Time horizon in years
        discount_rate : float, optional
            Discount rate for future losses
            
        Returns
        -------
        dict
            ECL and components
        """
        result = expected_credit_loss(pd, lgd, ead, time_horizon, discount_rate)
        self._history.append({"model": "expected_credit_loss", "result": result})
        return result
    
    def loan_pricing(self, loan_amount, term, pd, lgd, funding_cost, 
                    operating_cost, capital_requirement, target_roe):
        """
        Calculate risk-based loan pricing.
        
        Parameters
        ----------
        loan_amount : float
            Loan amount
        term : float
            Loan term in years
        pd : float
            Probability of default (annual)
        lgd : float
            Loss given default (as a decimal)
        funding_cost : float
            Cost of funds (annual rate)
        operating_cost : float
            Operating costs (as percentage of loan amount)
        capital_requirement : float
            Capital requirement as percentage of loan amount
        target_roe : float
            Target return on equity (annual rate)
            
        Returns
        -------
        dict
            Recommended interest rate and components
        """
        result = loan_pricing(
            loan_amount, term, pd, lgd, funding_cost, 
            operating_cost, capital_requirement, target_roe
        )
        self._history.append({"model": "loan_pricing", "result": result})
        return result
    
    def scoring_model_validation(self, predicted_scores, actual_defaults, score_bins=10):
        """
        Validate credit scoring model performance.
        
        Parameters
        ----------
        predicted_scores : array_like
            Predicted credit scores
        actual_defaults : array_like
            Actual default outcomes (0/1)
        score_bins : int, optional
            Number of score bins for analysis
            
        Returns
        -------
        dict
            Validation metrics (Gini, KS, AUC, etc.)
        """
        result = scoring_model_validation(predicted_scores, actual_defaults, score_bins)
        self._history.append({"model": "scoring_model_validation", "result": result})
        return result
    
    def loss_given_default(self, collateral_value, loan_amount, recovery_rate=None, 
                          liquidation_costs=0.1, time_to_recovery=1.0):
        """
        Estimate the loss given default for a loan.
        
        Parameters
        ----------
        collateral_value : float
            Value of collateral
        loan_amount : float
            Outstanding loan amount
        recovery_rate : float, optional
            Historical recovery rate for similar loans
        liquidation_costs : float, optional
            Costs associated with liquidating collateral
        time_to_recovery : float, optional
            Expected time to recovery in years
            
        Returns
        -------
        dict
            LGD estimate and components
        """
        result = loss_given_default(
            collateral_value, loan_amount, recovery_rate, 
            liquidation_costs, time_to_recovery
        )
        self._history.append({"model": "loss_given_default", "result": result})
        return result
    
    def exposure_at_default(self, current_balance, undrawn_amount, credit_conversion_factor=0.5):
        """
        Calculate exposure at default for credit facilities.
        
        Parameters
        ----------
        current_balance : float
            Current drawn balance
        undrawn_amount : float
            Undrawn commitment
        credit_conversion_factor : float, optional
            Factor to convert undrawn amounts to exposure
            
        Returns
        -------
        dict
            EAD and components
        """
        result = exposure_at_default(current_balance, undrawn_amount, credit_conversion_factor)
        self._history.append({"model": "exposure_at_default", "result": result})
        return result
    
    def get_history(self):
        """
        Get history of credit scoring calculations.
        
        Returns
        -------
        list
            History of calculations
        """
        return self._history 