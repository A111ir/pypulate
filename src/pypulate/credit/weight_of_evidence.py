"""
Weight of Evidence (WOE) and Information Value (IV) calculation.
"""

import numpy as np
from typing import Dict, Any, List, Union, cast
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
    good_arr = np.array(good_count, dtype=float)
    bad_arr = np.array(bad_count, dtype=float)
    
    if len(good_arr) == 0 or len(bad_arr) == 0:
        raise ValueError("Empty arrays are not allowed")
    
    if len(good_arr) != len(bad_arr):
        raise ValueError("Length of good_count and bad_count must match")
    
    total_good_count: float = float(np.sum(good_arr))
    total_bad_count: float = float(np.sum(bad_arr))
    total_samples: float = total_good_count + total_bad_count
    
    bin_samples = good_arr + bad_arr
    small_bins = bin_samples < (min_samples * total_samples)
    
    good_adj = np.copy(good_arr)
    bad_adj = np.copy(bad_arr)
    
    zero_good_mask = good_arr == 0
    zero_bad_mask = bad_arr == 0
    
    good_adj[zero_good_mask] += adjustment
    bad_adj[zero_bad_mask] += adjustment
    
    total_good_adj: float = float(np.sum(good_adj))
    total_bad_adj: float = float(np.sum(bad_adj))
    
    good_dist = good_adj / total_good_adj
    bad_dist = bad_adj / total_bad_adj
    
    woe = np.log(good_dist / bad_dist)
    iv: float = float(np.sum((good_dist - bad_dist) * woe))
    
    iv_strength: str
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