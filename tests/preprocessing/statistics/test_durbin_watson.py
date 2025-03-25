import pytest
import numpy as np
import warnings
from pypulate.preprocessing.statistics import durbin_watson_test

@pytest.fixture
def positive_autocorr_residuals():
    """Generate residuals with positive autocorrelation"""
    np.random.seed(42)
    n = 100
    e = np.random.normal(0, 1, n)
    residuals = np.zeros(n)
    residuals[0] = e[0]
    for i in range(1, n):
        residuals[i] = 0.7 * residuals[i-1] + e[i]
    return residuals

@pytest.fixture
def negative_autocorr_residuals():
    """Generate residuals with negative autocorrelation"""
    np.random.seed(42)
    n = 100
    residuals = np.zeros(n)
    for i in range(0, n, 2):
        if i+1 < n:
            val = np.random.normal(0, 1)
            residuals[i] = val
            residuals[i+1] = -val
    return residuals

@pytest.fixture
def no_autocorr_residuals():
    """Generate residuals with no autocorrelation"""
    np.random.seed(42)
    return np.random.normal(0, 1, 100)

def test_positive_autocorrelation(positive_autocorr_residuals):
    """Test that positive autocorrelation gives DW stat < 2"""
    dw = durbin_watson_test(positive_autocorr_residuals)
    assert isinstance(dw, float)
    assert 0 <= dw <= 2
    assert dw < 1.5  

def test_negative_autocorrelation(negative_autocorr_residuals):
    """Test that negative autocorrelation gives DW stat > 2"""
    dw = durbin_watson_test(negative_autocorr_residuals)
    assert isinstance(dw, float)
    assert 2 <= dw <= 4
    assert dw > 2.5  

def test_no_autocorrelation(no_autocorr_residuals):
    """Test that no autocorrelation gives DW stat ≈ 2"""
    dw = durbin_watson_test(no_autocorr_residuals)
    assert isinstance(dw, float)
    assert 1.5 <= dw <= 2.5  

def test_constant_residuals():
    """Test behavior with constant residuals"""
    residuals = np.ones(100)
    dw = durbin_watson_test(residuals)
    assert np.isnan(dw) 

def test_zero_residuals():
    """Test behavior with zero residuals"""
    residuals = np.zeros(100)
    dw = durbin_watson_test(residuals)
    assert np.isnan(dw) 

def test_single_value():
    """Test behavior with single value"""
    dw = durbin_watson_test([1.0])
    assert np.isnan(dw) 

def test_two_values():
    """Test behavior with two values"""
    dw = durbin_watson_test([1.0, 2.0])
    assert isinstance(dw, float)
    assert np.isfinite(dw)

def test_all_nan():
    """Test behavior with all NaN values"""
    residuals = np.full(100, np.nan)
    dw = durbin_watson_test(residuals)
    assert np.isnan(dw)  

def test_some_nan():
    """Test behavior with some NaN values"""
    np.random.seed(42)
    residuals = np.random.normal(0, 1, 35)  
    residuals[::2] = np.nan 
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always") 
        dw = durbin_watson_test(residuals)
        assert len(w) >= 1  
        assert any("less than 30" in str(warn.message) for warn in w)
    assert isinstance(dw, float)
    assert np.isfinite(dw)

def test_leading_trailing_nan():
    """Test behavior with NaN values at start and end"""
    np.random.seed(42)
    residuals = np.random.normal(0, 1, 40) 
    residuals[:5] = np.nan 
    residuals[-10:] = np.nan  
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  
        dw = durbin_watson_test(residuals)
        assert len(w) >= 1  
        assert any("less than 30" in str(warn.message) for warn in w)
    assert isinstance(dw, float)
    assert np.isfinite(dw)

def test_list_input():
    """Test that function accepts list input"""
    residuals = [1.0, 2.0, 3.0, 4.0]
    dw = durbin_watson_test(residuals)
    assert isinstance(dw, float)
    assert np.isfinite(dw)

def test_tuple_input():
    """Test that function accepts tuple input"""
    residuals = (1.0, 2.0, 3.0, 4.0)
    dw = durbin_watson_test(residuals)
    assert isinstance(dw, float)
    assert np.isfinite(dw)

def test_small_sample_warning():
    """Test warning for small sample size"""
    residuals = np.random.normal(0, 1, 20)
    with warnings.catch_warnings(record=True) as w:
        durbin_watson_test(residuals)
        assert len(w) == 1
        assert "less than 30" in str(w[0].message)

def test_large_input():
    """Test performance with large input"""
    residuals = np.random.normal(0, 1, 10000)
    dw = durbin_watson_test(residuals)
    assert isinstance(dw, float)
    assert np.isfinite(dw) 