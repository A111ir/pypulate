"""
Scorecard creation for credit scoring.
"""

from typing import Dict, Any, Optional


def create_scorecard(features: Dict[str, float], weights: Dict[str, float], 
                    offsets: Optional[Dict[str, float]]=None, 
                    scaling_factor: float=100.0, 
                    base_score: float=600) -> Dict[str, Any]:
    """
    Create a points-based scorecard for credit scoring.
    
    Parameters
    ----------
    features : dict
        Dictionary of feature names and their values (e.g., {"income": 75000, "age": 35}).
    weights : dict
        Dictionary of feature names and their weights (e.g., {"income": 0.3, "age": 0.5}).
    offsets : dict, optional
        Dictionary of feature names and offset values (default is 0 for each feature if None).
    scaling_factor : float, optional
        Scaling factor to divide the points, controlling the score range (default is 100.0).
    base_score : float, optional
        Base score to which feature points are added (default is 600).
        
    Returns
    -------
    dict
        Dictionary containing the total score, points breakdown, and risk category.
    """
    if offsets is None:
        offsets = {k: 0 for k in weights.keys()}
        
    points = {}
    total_points = base_score
    
    for feature, value in features.items():
        if feature in weights:
            feature_points = (value - offsets.get(feature, 0)) * weights[feature] / scaling_factor
            points[feature] = feature_points
            total_points += feature_points
    
    reference_scaling = 100.0
    reference_thresholds = {
        "Excellent": 750,
        "Good": 700,
        "Fair": 650,
        "Poor": 600
    }
    
    adjustment_factor = reference_scaling / scaling_factor
    thresholds = {
        category: base_score + (threshold - base_score) / adjustment_factor
        for category, threshold in reference_thresholds.items()
    }
    
    if total_points >= thresholds["Excellent"]:
        risk_category = "Excellent"
    elif total_points >= thresholds["Good"]:
        risk_category = "Good"
    elif total_points >= thresholds["Fair"]:
        risk_category = "Fair"
    elif total_points >= thresholds["Poor"]:
        risk_category = "Poor"
    else:
        risk_category = "Very Poor"
    
    result = {
        "total_score": total_points,
        "points_breakdown": points,
        "risk_category": risk_category,
        "thresholds": thresholds  
    }
    
    return result