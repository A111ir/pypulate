import numpy as np
import pytest
from pypulate.preprocessing import remove_outliers

def test_basic_zscore():
    """Test basic z-score outlier removal."""
    data = np.array([1, 2, 3, 100, 4, 5])
    result = remove_outliers(data, method='zscore', threshold=2.0)
    assert np.isnan(result[3])  # 100 should be marked as outlier
    assert not np.any(np.isnan(result[[0,1,2,4,5]]))  # Other values should remain

def test_basic_iqr():
    """Test basic IQR outlier removal."""
    data = np.array([1, 2, 3, 50, 4, 5])
    result = remove_outliers(data, method='iqr', threshold=1.5)
    assert np.isnan(result[3])  # 50 should be marked as outlier
    assert not np.any(np.isnan(result[[0,1,2,4,5]]))

def test_basic_mad():
    """Test basic MAD outlier removal."""
    data = np.array([1, 2, 3, 30, 4, 5])
    result = remove_outliers(data, method='mad', threshold=3.0)
    assert np.isnan(result[3])  # 30 should be marked as outlier
    assert not np.any(np.isnan(result[[0,1,2,4,5]]))

def test_no_outliers():
    """Test when there are no outliers."""
    data = np.array([1, 2, 3, 4, 5])
    for method in ['zscore', 'iqr', 'mad']:
        result = remove_outliers(data, method=method)
        np.testing.assert_array_equal(result, data)

def test_all_same_value():
    """Test when all values are the same."""
    data = np.array([1, 1, 1, 1])
    for method in ['zscore', 'iqr', 'mad']:
        result = remove_outliers(data, method=method)
        np.testing.assert_array_equal(result, data)

def test_empty_array():
    """Test with empty array."""
    data = np.array([])
    for method in ['zscore', 'iqr', 'mad']:
        result = remove_outliers(data, method=method)
        assert len(result) == 0

def test_with_nan():
    """Test with NaN values in input."""
    data = np.array([1, np.nan, 3, 100, 4, 5])
    result = remove_outliers(data, method='zscore', threshold=1.5)
    assert np.isnan(result[1])  # Original NaN
    assert np.isnan(result[3])  # 100 should be marked as outlier

def test_different_thresholds():
    """Test different threshold values."""
    data = np.array([1, 2, 3, 10, 4, 5])
    # Should be outlier with low threshold
    result1 = remove_outliers(data, threshold=1.0)
    assert np.isnan(result1[3])
    # Should not be outlier with high threshold
    result2 = remove_outliers(data, threshold=5.0)
    assert not np.isnan(result2[3])

def test_invalid_method():
    """Test invalid method raises ValueError."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        remove_outliers(data, method='invalid')

def test_negative_threshold():
    """Test negative threshold."""
    data = np.array([1, 2, 3, 4, 5])
    for method in ['zscore', 'iqr', 'mad']:
        result = remove_outliers(data, method=method, threshold=-1.0)
        assert np.all(np.isnan(result))  # All values should be marked as outliers

def test_extreme_values():
    """Test with extreme values."""
    data = np.array([1, 2, 3, 1e6, 4, 5])
    for method in ['zscore', 'iqr', 'mad']:
        result = remove_outliers(data, method=method)
        assert np.isnan(result[3])  # 1e6 should be marked as outlier

def test_asymmetric_outliers():
    """Test with asymmetric outliers."""
    data = np.array([1, 2, 3, 100, -100, 4, 5])
    result = remove_outliers(data, method='zscore', threshold=1.5)
    assert np.isnan(result[3])  # 100 should be marked as outlier
    assert np.isnan(result[4])  # -100 should be marked as outlier

def test_multiple_outliers():
    """Test with multiple outliers."""
    data = np.array([1, 2, 3, 100, 200, 4, 5])
    result = remove_outliers(data, method='zscore', threshold=1.5)
    assert np.isnan(result[3])  # 100 should be marked as outlier
    assert np.isnan(result[4])  # 200 should be marked as outlier

def test_different_input_types():
    """Test with different input types."""
    data_list = [1, 2, 3, 100, 4, 5]
    data_tuple = (1, 2, 3, 100, 4, 5)
    for data in [data_list, data_tuple]:
        result = remove_outliers(data)
        assert np.isnan(result[3])  # 100 should be marked as outlier

def test_floating_point_data():
    """Test with floating point data."""
    data = np.array([1.1, 2.2, 3.3, 100.5, 4.4, 5.5])
    result = remove_outliers(data)
    assert np.isnan(result[3])  # 100.5 should be marked as outlier