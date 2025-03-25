import numpy as np
import pytest
from pypulate.filters.signal_filters import (
    butterworth_filter,
    chebyshev_filter,
    savitzky_golay_filter,
    wiener_filter,
    median_filter,
    hampel_filter,
    hodrick_prescott_filter,
    baxter_king_filter
)

def test_butterworth_filter_shape():
    data = np.sin(np.linspace(0, 2 * np.pi, 100))
    filtered = butterworth_filter(data, cutoff=0.1, filter_type='lowpass', fs=100)
    assert filtered.shape == data.shape
    assert isinstance(filtered, np.ndarray)


def test_butterworth_lowpass_energy_reduction():
    # Create a signal with low and high frequency components.
    x = np.linspace(0, 1, 500)
    low_freq_component = np.sin(2 * np.pi * 5 * x)
    high_freq_component = 0.5 * np.sin(2 * np.pi * 50 * x)
    signal_data = low_freq_component + high_freq_component
    # Use a cutoff that should preserve the low frequency (5 Hz) and attenuate the high (50 Hz).
    filtered = butterworth_filter(signal_data, cutoff=10, filter_type='lowpass', fs=500)
    # The filtered signal should be highly correlated with the low frequency component.
    corr = np.corrcoef(filtered, low_freq_component)[0, 1]
    assert corr > 0.9


def test_chebyshev_filter_shape():
    data = np.sin(np.linspace(0, 2 * np.pi, 100))
    filtered = chebyshev_filter(data, cutoff=0.1, ripple=0.5, filter_type='lowpass', fs=100, type_num=1)
    assert filtered.shape == data.shape
    assert isinstance(filtered, np.ndarray)


def test_savitzky_golay_filter_even_window_adjustment():
    # Provide an even window length, which should be adjusted to an odd number internally.
    data = np.arange(15, dtype=float)
    filtered = savitzky_golay_filter(data, window_length=10, polyorder=2)
    assert filtered.shape == data.shape


def test_wiener_filter_shape():
    data = np.random.randn(100)
    filtered = wiener_filter(data, mysize=5)
    assert filtered.shape == data.shape


def test_median_filter_outlier_replacement():
    # Create an array with an outlier in the middle.
    data = np.array([1, 1, 100, 1, 1], dtype=float)
    filtered = median_filter(data, kernel_size=3)
    # The outlier at index 2 should be replaced by the median (which is 1).
    np.testing.assert_equal(filtered[2], 1)


def test_hampel_filter_outlier_replacement():
    # Create a signal with a clear outlier.
    data = np.array([1, 1, 100, 1, 1], dtype=float)
    # Use a window size that covers most of the values and a low threshold to force replacement.
    filtered = hampel_filter(data, window_size=2, n_sigmas=1)
    # The outlier should be replaced with the median value (1).
    np.testing.assert_equal(filtered[2], 1)


def test_hodrick_prescott_filter_reconstruction():
    x = np.linspace(0, 1, 50)
    data = x + np.sin(2 * np.pi * x)
    trend, cycle = hodrick_prescott_filter(data, lambda_param=1600)
    np.testing.assert_allclose(trend + cycle, data, rtol=1e-5)


def test_baxter_king_filter_nan_boundaries():
    data = np.sin(np.linspace(0, 2 * np.pi, 100))
    filtered = baxter_king_filter(data, low=6, high=32, K=12)
    # Check that the first 12 and the last 12 values are NaN.
    assert np.all(np.isnan(filtered[:12])), "First 12 values should be NaN"
    assert np.all(np.isnan(filtered[-12:])), "Last 12 values should be NaN"
    # Ensure that not all values are NaN (i.e., the middle values are computed).
    assert not np.all(np.isnan(filtered)), "Non-boundary values should not be NaN"


def test_input_conversion_to_numpy():
    # Test that if data is provided as a list, it is converted to a numpy array.
    data_list = [1, 2, 3, 4, 5]
    filtered = wiener_filter(data_list, mysize=3)
    assert isinstance(filtered, np.ndarray) 