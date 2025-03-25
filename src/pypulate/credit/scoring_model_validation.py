"""
Credit scoring model validation.
"""

import numpy as np
from typing import Dict, Any, List
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
    
    # For credit scoring, lower scores should indicate higher default risk
    # So non-defaults (0s) should have higher scores, and defaults (1s) should have lower scores
    # in a well-performing model. We need to invert the scores for ROC calculation.
    
    # Initialize empty lists for ROC curve data
    tpr: List[float] = []
    fpr: List[float] = []
    thresholds: List[float] = []
    
    # Count positives and negatives
    n_pos : float = np.sum(actual_defaults)
    n_neg = len(actual_defaults) - n_pos
    
    # Check if all scores are constant
    if np.all(predicted_scores == predicted_scores[0]):
        # For constant scores, AUC should be 0.5 (random classifier)
        auc = 0.5
        tpr = [0.0, 1.0]
        fpr = [0.0, 1.0]
        thresholds = [float(predicted_scores[0] + 0.001), float(predicted_scores[0] - 0.001)]
    elif n_pos == 0 or n_neg == 0:
        # Handle the case where all examples are of one class
        auc = 0.5  # Default to random classifier performance
        tpr = [0.0, 1.0]
        fpr = [0.0, 1.0]
        thresholds = [float(np.max(predicted_scores) + 0.001), float(np.min(predicted_scores) - 0.001)]
    else:
        # Sort scores and defaults by score
        sorted_indices = np.argsort(predicted_scores)
        sorted_scores = predicted_scores[sorted_indices]
        sorted_defaults = actual_defaults[sorted_indices]
        
        # Calculate ROC curve for defaults (lower scores should indicate higher probability of default)
        # So we're measuring if defaults (1s) are concentrated at the lower score range
        tpr = [0.0]  # True positive rate
        fpr = [0.0]  # False positive rate
        thresholds = [float(np.max(predicted_scores) + 0.001)]  # Start with a threshold higher than any score
        
        tp_count = 0
        fp_count = 0
        prev_score = None
        
        # Iterate through scores from lowest to highest
        for i, (score, default) in enumerate(zip(sorted_scores, sorted_defaults)):
            # Only add a point if the score changes or we're at the last point
            if prev_score is not None and score != prev_score or i == len(sorted_scores) - 1:
                tpr.append(tp_count / n_pos if n_pos > 0 else 0)
                fpr.append(fp_count / n_neg if n_neg > 0 else 0)
                thresholds.append(float(score))
            
            # Update counts
            if default == 1:  # Actual default
                tp_count += 1
            else:  # Non-default
                fp_count += 1
                
            prev_score = score
            
        # Add the final point (1,1)
        if tpr[-1] < 1.0 or fpr[-1] < 1.0:
            tpr.append(1.0)
            fpr.append(1.0)
            thresholds.append(float(np.min(predicted_scores) - 0.001))  # Threshold lower than any score
        
        # Calculate AUC using the trapezoidal rule
        auc = 0.0
        for i in range(1, len(tpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    
    # Gini coefficient
    gini = 2 * auc - 1
    
    # KS statistic
    ks_stat = float(np.max(np.abs(np.array(tpr) - np.array(fpr))))
    
    # Create score bins for analysis
    bin_edges = np.array(np.percentile(predicted_scores, np.linspace(0, 100, score_bins + 1)))
    
    # Handle edge case where all scores are identical, making bin_edges a scalar
    if np.isscalar(bin_edges) or len(bin_edges) <= 1:
        bin_edges = np.array([float(predicted_scores.min()) - 0.001, float(predicted_scores.max()) + 0.001])
        bin_indices = np.zeros(len(predicted_scores), dtype=int)
    else:
        bin_indices = np.digitize(predicted_scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, score_bins - 1)
    
    bin_counts = []
    bin_defaults = []
    bin_default_rates = []
    
    for i in range(score_bins):
        bin_mask = bin_indices == i
        count = int(np.sum(bin_mask))
        defaults = int(np.sum(actual_defaults[bin_mask]))
        
        bin_counts.append(count)
        bin_defaults.append(defaults)
        bin_default_rates.append(defaults / count if count > 0 else 0)
    
    good_dist = []
    bad_dist = []
    
    for i in range(score_bins):
        good_count = bin_counts[i] - bin_defaults[i]
        bad_count = bin_defaults[i]
        
        total_good = int(np.sum(np.array(bin_counts) - np.array(bin_defaults)))
        total_bad = int(np.sum(bin_defaults))
        
        good_dist.append(good_count / total_good if total_good > 0 else 0)
        bad_dist.append(bad_count / total_bad if total_bad > 0 else 0)
    
    adjustment = 0.0001
    good_dist_array = np.array(good_dist) + adjustment * (np.array(good_dist) == 0)
    bad_dist_array = np.array(bad_dist) + adjustment * (np.array(bad_dist) == 0)
    
    woe = np.log(good_dist_array / bad_dist_array)
    iv = float(np.sum((good_dist_array - bad_dist_array) * woe))
    
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
    for i in range(len(bin_edges) - 1): 
        bin_info.append({
            "bin": i + 1,
            "min_score": float(bin_edges[i]),
            "max_score": float(bin_edges[i + 1]),
            "count": bin_counts[i] if i < len(bin_counts) else 0,
            "defaults": bin_defaults[i] if i < len(bin_defaults) else 0,
            "default_rate": bin_default_rates[i] if i < len(bin_default_rates) else 0,
            "woe": float(woe[i]) if i < len(woe) else 0.0
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
            "thresholds": thresholds
        },
        "bin_analysis": bin_info
    }
    
    return result 