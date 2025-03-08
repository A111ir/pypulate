"""
Credit scoring model validation.
"""

import numpy as np
from typing import Dict, Any
from numpy.typing import ArrayLike


def scoring_model_validation(predicted_scores: ArrayLike, actual_defaults: ArrayLike, 
                            score_bins: int=10) -> Dict[str, Any]:
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
    predicted_scores = np.array(predicted_scores)
    actual_defaults = np.array(actual_defaults)
    
    if len(predicted_scores) != len(actual_defaults):
        raise ValueError("Length of predicted_scores and actual_defaults must match")
    if not np.all(np.isin(actual_defaults, [0, 1])):
        raise ValueError("actual_defaults must contain only 0 and 1 values")
    
    thresholds = np.linspace(np.min(predicted_scores), np.max(predicted_scores), 100)
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        predicted_positive = predicted_scores >= threshold
        tp = np.sum((predicted_positive) & (actual_defaults == 1))
        fp = np.sum((predicted_positive) & (actual_defaults == 0))
        tn = np.sum((~predicted_positive) & (actual_defaults == 0))
        fn = np.sum((~predicted_positive) & (actual_defaults == 1))
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    auc = 0.0 
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    
    gini = 2 * auc - 1
    
    ks_stat = np.max(np.abs(np.array(tpr) - np.array(fpr)))
    
    bin_edges = np.percentile(predicted_scores, np.linspace(0, 100, score_bins + 1))
    bin_indices = np.digitize(predicted_scores, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, score_bins - 1)
    
    bin_counts = []
    bin_defaults = []
    bin_default_rates = []
    
    for i in range(score_bins):
        bin_mask = bin_indices == i
        count = np.sum(bin_mask)
        defaults = np.sum(actual_defaults[bin_mask])
        
        bin_counts.append(count)
        bin_defaults.append(defaults)
        bin_default_rates.append(defaults / count if count > 0 else 0)
    
    good_dist = []
    bad_dist = []
    
    for i in range(score_bins):
        good_count = bin_counts[i] - bin_defaults[i]
        bad_count = bin_defaults[i]
        
        total_good = np.sum(np.array(bin_counts) - np.array(bin_defaults))
        total_bad = np.sum(bin_defaults)
        
        good_dist.append(good_count / total_good if total_good > 0 else 0)
        bad_dist.append(bad_count / total_bad if total_bad > 0 else 0)
    
    adjustment = 0.0001
    good_dist_array = np.array(good_dist) + adjustment * (np.array(good_dist) == 0)
    bad_dist_array = np.array(bad_dist) + adjustment * (np.array(bad_dist) == 0)
    
    woe = np.log(good_dist_array / bad_dist_array)
    iv = np.sum((good_dist_array - bad_dist_array) * woe)
    
    concordant_pairs = 0
    discordant_pairs = 0
    tied_pairs = 0
    
    max_pairs = 10000
    if len(predicted_scores) > 1000:
        # Sample pairs
        np.random.seed(42)
        indices = np.random.choice(len(predicted_scores), size=min(1000, len(predicted_scores)), replace=False)
        sampled_scores = predicted_scores[indices]
        sampled_defaults = actual_defaults[indices]
    else:
        sampled_scores = predicted_scores
        sampled_defaults = actual_defaults
    
    default_indices = np.where(sampled_defaults == 1)[0]
    non_default_indices = np.where(sampled_defaults == 0)[0]
    
    pair_count = 0
    for i in default_indices:
        for j in non_default_indices:
            pair_count += 1
            if pair_count > max_pairs:
                break
                
            if sampled_scores[i] < sampled_scores[j]:
                concordant_pairs += 1
            elif sampled_scores[i] > sampled_scores[j]:
                discordant_pairs += 1
            else:
                tied_pairs += 1
        
        if pair_count > max_pairs:
            break
    
    total_pairs = concordant_pairs + discordant_pairs + tied_pairs
    concordance = concordant_pairs / total_pairs if total_pairs > 0 else 0
    
    bin_info = []
    for i in range(score_bins):
        bin_info.append({
            "bin": i + 1,
            "min_score": bin_edges[i],
            "max_score": bin_edges[i + 1],
            "count": bin_counts[i],
            "defaults": bin_defaults[i],
            "default_rate": bin_default_rates[i],
            "woe": woe[i]
        })
    
    result = {
        "auc": auc,
        "gini": gini,
        "ks_statistic": ks_stat,
        "information_value": iv,
        "concordance": concordance,
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds.tolist()
        },
        "bin_analysis": bin_info
    }
    
    return result 