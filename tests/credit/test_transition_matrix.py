import pytest
import numpy as np
from pypulate.credit.transition_matrix import transition_matrix


def test_basic_transition():
    ratings_t0 = ["A", "A", "B", "B", "C"]
    ratings_t1 = ["A", "B", "B", "C", "C"]
    
    result = transition_matrix(ratings_t0, ratings_t1)
    
    assert "transition_matrix" in result
    assert "probability_matrix" in result
    assert "ratings" in result
    
    # Verify ratings list
    assert sorted(result["ratings"]) == ["A", "B", "C"]
    
    # Check transition counts (from -> to)
    # A->A: 1, A->B: 1, A->C: 0
    # B->A: 0, B->B: 1, B->C: 1
    # C->A: 0, C->B: 0, C->C: 1
    trans_mat = np.array(result["transition_matrix"])
    assert trans_mat[0, 0] == 1  # A->A
    assert trans_mat[0, 1] == 1  # A->B
    assert trans_mat[1, 1] == 1  # B->B
    assert trans_mat[1, 2] == 1  # B->C
    assert trans_mat[2, 2] == 1  # C->C
    
    # Check probability matrix
    prob_mat = np.array(result["probability_matrix"])
    assert prob_mat[0, 0] == 0.5  # A->A: 50%
    assert prob_mat[0, 1] == 0.5  # A->B: 50%
    assert prob_mat[1, 1] == 0.5  # B->B: 50%
    assert prob_mat[1, 2] == 0.5  # B->C: 50%
    assert prob_mat[2, 2] == 1.0  # C->C: 100%


def test_numeric_ratings():
    ratings_t0 = [1, 2, 2, 3, 3, 3]
    ratings_t1 = [1, 1, 2, 2, 3, 4]
    
    result = transition_matrix(ratings_t0, ratings_t1)
    
    # Verify ratings list
    assert sorted(result["ratings"]) == [1, 2, 3, 4]
    
    # Check transition matrix
    trans_mat = np.array(result["transition_matrix"])
    assert trans_mat[0, 0] == 1  # 1->1
    assert trans_mat[1, 0] == 1  # 2->1
    assert trans_mat[1, 1] == 1  # 2->2
    assert trans_mat[2, 1] == 1  # 3->2
    assert trans_mat[2, 2] == 1  # 3->3
    assert trans_mat[2, 3] == 1  # 3->4
    
    # Check probability matrix
    prob_mat = np.array(result["probability_matrix"])
    assert prob_mat[0, 0] == 1.0  # 1->1: 100%
    assert prob_mat[1, 0] == 0.5  # 2->1: 50%
    assert prob_mat[1, 1] == 0.5  # 2->2: 50%
    assert prob_mat[2, 1] == 1/3  # 3->2: 33.3%
    assert prob_mat[2, 2] == 1/3  # 3->3: 33.3%
    assert prob_mat[2, 3] == 1/3  # 3->4: 33.3%


def test_empty_arrays():
    ratings_t0 = []
    ratings_t1 = []
    
    result = transition_matrix(ratings_t0, ratings_t1)
    
    assert result["transition_matrix"] == []
    assert result["probability_matrix"] == []
    assert result["ratings"] == []


def test_single_rating():
    ratings_t0 = ["AAA"]
    ratings_t1 = ["AA"]
    
    result = transition_matrix(ratings_t0, ratings_t1)
    
    # Verify ratings list
    assert sorted(result["ratings"]) == ["AA", "AAA"]
    
    # Check matrices
    trans_mat = np.array(result["transition_matrix"])
    # The sort order matters, we need to find the correct indices
    ratings = result["ratings"]
    aaa_idx = ratings.index("AAA")
    aa_idx = ratings.index("AA")
    
    assert trans_mat[aaa_idx, aa_idx] == 1  # AAA->AA
    
    prob_mat = np.array(result["probability_matrix"])
    assert prob_mat[aaa_idx, aa_idx] == 1.0  # AAA->AA: 100%


def test_mixed_type_inputs():
    ratings_t0 = np.array(["A", "B", "C"])
    ratings_t1 = ["A", "A", "B"]
    
    result = transition_matrix(ratings_t0, ratings_t1)
    
    # Check matrices
    trans_mat = np.array(result["transition_matrix"])
    assert trans_mat[0, 0] == 1  # A->A
    assert trans_mat[1, 0] == 1  # B->A
    assert trans_mat[2, 1] == 1  # C->B


def test_lists_equal_length():
    # Test with mismatched array lengths
    ratings_t0 = ["A", "B", "C"]
    ratings_t1 = ["A", "B"]
    
    with pytest.raises(IndexError):
        transition_matrix(ratings_t0, ratings_t1)


def test_no_transitions():
    # All ratings stay the same
    ratings_t0 = ["A", "B", "C", "D"]
    ratings_t1 = ["A", "B", "C", "D"]
    
    result = transition_matrix(ratings_t0, ratings_t1)
    
    # Check probability matrix - should be identity matrix
    prob_mat = np.array(result["probability_matrix"])
    assert np.all(np.diag(prob_mat) == 1.0)
    assert np.sum(prob_mat) == 4.0  # Sum of all probabilities = number of ratings


def test_full_transition():
    # All ratings change
    ratings_t0 = ["A", "B", "C"]
    ratings_t1 = ["B", "C", "A"]
    
    result = transition_matrix(ratings_t0, ratings_t1)
    
    # Check transition counts
    trans_mat = np.array(result["transition_matrix"])
    assert trans_mat[0, 1] == 1  # A->B
    assert trans_mat[1, 2] == 1  # B->C
    assert trans_mat[2, 0] == 1  # C->A
    
    # Check probability matrix
    prob_mat = np.array(result["probability_matrix"])
    assert prob_mat[0, 1] == 1.0  # A->B: 100%
    assert prob_mat[1, 2] == 1.0  # B->C: 100%
    assert prob_mat[2, 0] == 1.0  # C->A: 100% 