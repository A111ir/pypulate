"""
Weight of Evidence (WOE) and Information Value (IV) calculation.
"""

import numpy as np
from typing import Dict, Any
from numpy.typing import ArrayLike


def weight_of_evidence(good_count: ArrayLike, bad_count: ArrayLike, 
                      min_samples: float=0.01, adjustment: float=0.5) -> Dict[str, Any]:
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
    good_count = np.array(good_count)
    bad_count = np.array(bad_count)
    
    total_samples = np.sum(good_count) + np.sum(bad_count)
    bin_samples = good_count + bad_count
    small_bins = bin_samples < (min_samples * total_samples)
    
    good_count = good_count + adjustment * (good_count == 0)
    bad_count = bad_count + adjustment * (bad_count == 0)
    
    total_good = np.sum(good_count)
    total_bad = np.sum(bad_count)
    
    good_dist = good_count / total_good
    bad_dist = bad_count / total_bad
    
    woe = np.log(good_dist / bad_dist)
    
    iv = np.sum((good_dist - bad_dist) * woe)
    
    if iv < 0.02:
        iv_strength = "Not predictive"
    elif iv < 0.1:
        iv_strength = "Weak predictive power"
    elif iv < 0.3:
        iv_strength = "Medium predictive power"
    elif iv < 0.5:
        iv_strength = "Strong predictive power"
    else:
        iv_strength = "Very strong predictive power"
    
    result = {
        "woe": woe.tolist(),
        "information_value": iv,
        "iv_strength": iv_strength,
        "good_distribution": good_dist.tolist(),
        "bad_distribution": bad_dist.tolist(),
        "small_bins": small_bins.tolist()  
    }
    
    return result 