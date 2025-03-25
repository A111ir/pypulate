import pytest
import numpy as np
from pypulate.credit.scoring_model_validation import scoring_model_validation

# Basic functionality tests
def test_basic_validation():
    """Test the basic functionality of the scoring model validation"""
    # Create simple test data
    predicted_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    actual_defaults = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # Check that the result contains all expected metrics
    assert isinstance(result, dict)
    assert "auc" in result
    assert "gini" in result
    assert "ks_statistic" in result
    assert "information_value" in result
    assert "concordance" in result
    assert "roc_curve" in result
    assert "bin_analysis" in result
    
    # For a perfect model with this data, AUC should be 1.0 (or very close)
    assert result["auc"] > 0.95
    
    # Perfect Gini should be 1.0 (or very close)
    assert result["gini"] > 0.95
    
    # KS statistic should be 1.0 for perfect separation
    assert result["ks_statistic"] > 0.95
    
    # Check ROC curve data structure
    assert "fpr" in result["roc_curve"]
    assert "tpr" in result["roc_curve"]
    assert "thresholds" in result["roc_curve"]
    
    # Check bin analysis
    assert len(result["bin_analysis"]) == 10  # Default is 10 bins

def test_perfect_separation():
    """Test a case of perfect separation between defaults and non-defaults"""
    # Create perfectly separated scores
    np.random.seed(42)
    non_default_scores = np.random.uniform(0.7, 1.0, 50)
    default_scores = np.random.uniform(0.0, 0.3, 50)
    
    predicted_scores = np.concatenate([non_default_scores, default_scores])
    actual_defaults = np.concatenate([np.zeros(50), np.ones(50)])
    
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # For perfect separation, metrics should be close to 1
    assert result["auc"] > 0.95
    assert result["gini"] > 0.90  # Gini = 2*AUC - 1
    assert result["ks_statistic"] > 0.95
    assert result["concordance"] > 0.95

def test_random_model():
    """Test a model with no predictive power (random predictions)"""
    np.random.seed(42)
    predicted_scores = np.random.uniform(0, 1, 100)
    actual_defaults = np.random.randint(0, 2, 100)
    
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # For a random model, metrics should be close to 0.5 or 0
    assert abs(result["auc"] - 0.5) < 0.15
    assert abs(result["gini"]) < 0.3  # Gini = 2*AUC - 1 ≈ 0
    assert result["ks_statistic"] < 0.3  # Should be low for random predictions

def test_inverted_model():
    """Test a model with inverted predictions (higher scores for defaults)"""
    # Create inverted scores (higher scores for defaults)
    np.random.seed(42)
    non_default_scores = np.random.uniform(0.0, 0.3, 50)
    default_scores = np.random.uniform(0.7, 1.0, 50)
    
    predicted_scores = np.concatenate([non_default_scores, default_scores])
    actual_defaults = np.concatenate([np.zeros(50), np.ones(50)])
    
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # For inverted model, AUC should be close to 0
    assert result["auc"] < 0.1
    assert result["gini"] < -0.8  # Gini = 2*AUC - 1 ≈ -1
    
    # Re-run with swapped scores to check consistency
    result_swapped = scoring_model_validation(1 - predicted_scores, actual_defaults)
    assert result_swapped["auc"] > 0.9
    assert result_swapped["gini"] > 0.8

def test_custom_bins():
    """Test with a custom number of score bins"""
    predicted_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    actual_defaults = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    result = scoring_model_validation(predicted_scores, actual_defaults, score_bins=5)
    
    # Check bin count
    assert len(result["bin_analysis"]) == 5
    
    # Verify bin properties
    for bin_info in result["bin_analysis"]:
        assert "bin" in bin_info
        assert "min_score" in bin_info
        assert "max_score" in bin_info
        assert "count" in bin_info
        assert "defaults" in bin_info
        assert "default_rate" in bin_info
        assert "woe" in bin_info
        
    # Check bin ordering (default rates should generally decrease with higher bins)
    default_rates = [bin_info["default_rate"] for bin_info in result["bin_analysis"]]
    assert default_rates[0] >= default_rates[-1]  # First bin should have higher default rate than last

def test_real_world_like_data():
    """Test with more realistic dataset"""
    np.random.seed(42)
    sample_size = 1000
    
    # Generate scores with some correlation to defaults
    # Good borrowers (no default) have scores centered around 700
    # Bad borrowers (default) have scores centered around 550
    good_scores = np.random.normal(700, 80, int(sample_size * 0.9))
    bad_scores = np.random.normal(550, 100, int(sample_size * 0.1))
    
    # Normalize to 0-1 range for easier interpretation
    all_scores = np.concatenate([good_scores, bad_scores])
    min_score, max_score = np.min(all_scores), np.max(all_scores)
    predicted_scores = (all_scores - min_score) / (max_score - min_score)
    
    # Create corresponding default flags (0 for good, 1 for bad)
    actual_defaults = np.concatenate([np.zeros(len(good_scores)), np.ones(len(bad_scores))])
    
    # Shuffle the data to mix good and bad borrowers
    indices = np.arange(len(predicted_scores))
    np.random.shuffle(indices)
    predicted_scores = predicted_scores[indices]
    actual_defaults = actual_defaults[indices]
    
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # For a reasonably good model, metrics should be decent
    assert 0.7 < result["auc"] < 0.95
    assert 0.4 < result["gini"] < 0.9
    assert 0.4 < result["ks_statistic"] < 0.9
    assert result["information_value"] > 0.5  # IV > 0.5 is considered strong
    
    # Check monotonicity of default rates across bins
    # Default rate should generally decrease as score bin increases
    default_rates = []
    for bin_info in result["bin_analysis"]:
        if bin_info["count"] > 0:  # Only consider bins with data
            default_rates.append(bin_info["default_rate"])
    
    # Check if default rates are mostly decreasing
    decreasing_pairs = sum(default_rates[i] >= default_rates[i+1] 
                          for i in range(len(default_rates)-1))
    assert decreasing_pairs >= 0.7 * (len(default_rates) - 1)  # At least 70% of adjacent bins should show decreasing default rates

# Input validation tests
def test_input_length_validation():
    """Test validation of input array lengths"""
    predicted_scores = np.array([0.1, 0.2, 0.3, 0.4])
    actual_defaults = np.array([1, 0, 1])  # One element short
    
    with pytest.raises(ValueError, match="Length of predicted_scores and actual_defaults must match"):
        scoring_model_validation(predicted_scores, actual_defaults)

def test_actual_defaults_validation():
    """Test validation of actual_defaults values"""
    predicted_scores = np.array([0.1, 0.2, 0.3, 0.4])
    actual_defaults = np.array([1, 0, 1, 2])  # Contains invalid value 2
    
    with pytest.raises(ValueError, match="actual_defaults must contain only 0 and 1 values"):
        scoring_model_validation(predicted_scores, actual_defaults)

def test_empty_inputs():
    """Test with empty input arrays"""
    predicted_scores = np.array([])
    actual_defaults = np.array([])
    
    with pytest.raises(Exception):  # Should raise some kind of error
        scoring_model_validation(predicted_scores, actual_defaults)

def test_single_class():
    """Test with only one class in actual_defaults"""
    # Only non-defaults
    predicted_scores = np.array([0.1, 0.2, 0.3, 0.4])
    actual_defaults = np.array([0, 0, 0, 0])
    
    # This should run without errors, but some metrics will be undefined
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # Only defaults
    predicted_scores = np.array([0.1, 0.2, 0.3, 0.4])
    actual_defaults = np.array([1, 1, 1, 1])
    
    # This should run without errors, but some metrics will be undefined
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # Both cases should return a complete result dict
    assert isinstance(result, dict)
    assert "auc" in result
    assert "gini" in result

def test_constant_scores():
    """Test with constant predicted scores"""
    predicted_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    actual_defaults = np.array([1, 0, 1, 0, 1])
    
    # This should run without errors
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # For constant scores, AUC should be 0.5
    assert abs(result["auc"] - 0.5) < 0.1
    assert abs(result["gini"]) < 0.2

def test_input_types():
    """Test different input types"""
    # Test with lists
    predicted_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    actual_defaults = [1, 1, 0, 0, 0]
    
    result_lists = scoring_model_validation(predicted_scores, actual_defaults)
    assert isinstance(result_lists, dict)
    
    # Test with numpy arrays
    result_arrays = scoring_model_validation(np.array(predicted_scores), np.array(actual_defaults))
    assert isinstance(result_arrays, dict)
    
    # Results should be the same
    assert abs(result_lists["auc"] - result_arrays["auc"]) < 1e-10
    assert abs(result_lists["gini"] - result_arrays["gini"]) < 1e-10

def test_large_dataset_performance():
    """Test performance with a large dataset"""
    np.random.seed(42)
    sample_size = 10000
    
    # Generate scores with some correlation to defaults
    good_scores = np.random.normal(0.7, 0.15, int(sample_size * 0.9))
    bad_scores = np.random.normal(0.4, 0.15, int(sample_size * 0.1))
    
    # Clip values to 0-1 range
    good_scores = np.clip(good_scores, 0, 1)
    bad_scores = np.clip(bad_scores, 0, 1)
    
    predicted_scores = np.concatenate([good_scores, bad_scores])
    actual_defaults = np.concatenate([np.zeros(len(good_scores)), np.ones(len(bad_scores))])
    
    # Shuffle the data
    indices = np.arange(len(predicted_scores))
    np.random.shuffle(indices)
    predicted_scores = predicted_scores[indices]
    actual_defaults = actual_defaults[indices]
    
    # This should run without errors and in a reasonable time
    result = scoring_model_validation(predicted_scores, actual_defaults)
    
    # Check the results are reasonable
    assert 0.7 < result["auc"] < 0.95
    assert 0.4 < result["gini"] < 0.9 