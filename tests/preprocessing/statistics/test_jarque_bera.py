import numpy as np
import pytest
from pypulate.preprocessing.statistics import jarque_bera_test

@pytest.fixture
def normal_data():
    np.random.seed(42)
    return np.random.normal(0, 1, 1000)  # Standard normal distribution

@pytest.fixture
def uniform_data():
    np.random.seed(42)
    return np.random.uniform(-1, 1, 1000)  # Uniform distribution (non-normal)

@pytest.fixture
def skewed_data():
    np.random.seed(42)
    # Log-normal distribution (right-skewed)
    return np.exp(np.random.normal(0, 0.5, 1000))

@pytest.fixture
def heavy_tailed_data():
    np.random.seed(42)
    # Student's t-distribution with 3 degrees of freedom (heavy tails)
    return np.random.standard_t(df=3, size=1000)

def test_normal_distribution(normal_data):
    stat, p_value = jarque_bera_test(normal_data)
    assert p_value > 0.05  # Should not reject normality

def test_uniform_distribution(uniform_data):
    stat, p_value = jarque_bera_test(uniform_data)
    assert p_value < 0.05  # Should reject normality

def test_skewed_distribution(skewed_data):
    stat, p_value = jarque_bera_test(skewed_data)
    assert p_value < 0.05  # Should reject normality

def test_heavy_tailed_distribution(heavy_tailed_data):
    stat, p_value = jarque_bera_test(heavy_tailed_data)
    assert p_value < 0.05  # Should reject normality

def test_constant_data():
    constant_data = np.ones(100)
    stat, p_value = jarque_bera_test(constant_data)
    assert np.isinf(stat)  # Expect infinite test statistic for constant data
    assert p_value == 0.0  # Strong evidence against normality

def test_insufficient_data():
    data = np.array([1, 2])
    stat, p_value = jarque_bera_test(data)
    assert np.isnan(stat) and np.isnan(p_value)  # Not enough data points

def test_nan_handling():
    data = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 20)
    stat, p_value = jarque_bera_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_infinite_values():
    with pytest.warns(RuntimeWarning):
        data = np.array([1.0, np.inf, 3.0, -np.inf, 5.0] * 20)
        stat, p_value = jarque_bera_test(data)
        assert np.isinf(stat)  # Expect infinite test statistic
        assert p_value == 0.0  # Strong evidence against normality

def test_small_sample_warning():
    data = np.random.randn(20)
    with pytest.warns(RuntimeWarning):
        jarque_bera_test(data)

def test_mixed_data_types():
    data = np.array([1, 2.5, 3, 4.5, 5] * 20)
    stat, p_value = jarque_bera_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_large_values():
    data = np.random.normal(1e6, 1e3, 1000)
    stat, p_value = jarque_bera_test(data)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

def test_zero_variance():
    # Data with extremely small but non-zero variance
    data = np.ones(1000)  # Start with constant data
    data[0] = 1.1  # Add a single outlier
    stat, p_value = jarque_bera_test(data)
    assert stat > 0  # Should return a finite positive test statistic
    assert p_value < 0.05  # Should reject normality for highly skewed data 