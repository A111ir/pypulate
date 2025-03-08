"""
Credit rating transition matrix calculation.
"""

import numpy as np
from typing import Dict, Any, List, Union
from numpy.typing import ArrayLike


def transition_matrix(ratings_t0: ArrayLike, ratings_t1: ArrayLike) -> Dict[str, Any]:
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
    ratings_t0 = np.array(ratings_t0)
    ratings_t1 = np.array(ratings_t1)
    
    unique_ratings = np.unique(np.concatenate([ratings_t0, ratings_t1]))
    n_ratings = len(unique_ratings)
    
    rating_to_idx = {rating: i for i, rating in enumerate(unique_ratings)}
    
    trans_matrix = np.zeros((n_ratings, n_ratings))
    
    for i in range(len(ratings_t0)):
        idx_t0 = rating_to_idx[ratings_t0[i]]
        idx_t1 = rating_to_idx[ratings_t1[i]]
        trans_matrix[idx_t0, idx_t1] += 1
    
    row_sums = trans_matrix.sum(axis=1)
    prob_matrix = np.zeros_like(trans_matrix)
    
    for i in range(n_ratings):
        if row_sums[i] > 0:
            prob_matrix[i, :] = trans_matrix[i, :] / row_sums[i]
    
    result = {
        "transition_matrix": trans_matrix.tolist(),
        "probability_matrix": prob_matrix.tolist(),
        "ratings": unique_ratings.tolist()
    }
    
    return result