import os
import numpy as np
import pandas as pd
import pytest

from pypulate.transforms import wave, zigzag


# Create a fixture for sample data
@pytest.fixture
def sample_ohlc_data():
    """Fixture providing sample OHLC data for testing."""
    # Generate sample OHLC data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Create price data with some patterns
    close = np.array([100.0 + 10 * np.sin(i / 10) + i / 10 for i in range(100)])
    open_prices = np.array([close[i-1] if i > 0 else 100.0 for i in range(100)])
    high_prices = np.array([max(open_prices[i], close[i]) + np.random.rand() * 2 for i in range(100)])
    low_prices = np.array([min(open_prices[i], close[i]) - np.random.rand() * 2 for i in range(100)])
    
    return {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close,
        'dates': dates
    }

@pytest.fixture
def sample_price_data():
    """Fixture providing sample price data for testing."""
    # Generate sample price data with some patterns
    return np.array([100.0 + 10 * np.sin(i / 10) + i / 10 for i in range(100)])


class TestWaveFunction:
    """Test suite for the wave function."""
    
    def test_wave_with_all_parameters(self, sample_ohlc_data):
        """Test wave function with all parameters provided."""
        result = wave(
            open=sample_ohlc_data['open'],
            high=sample_ohlc_data['high'],
            low=sample_ohlc_data['low'],
            close=sample_ohlc_data['close']
        )
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1  # Wave function returns a 1D array
        assert len(result) > 0  # Should have some wave points
        
        # Values should be within the range of prices
        min_price = min(
            np.min(sample_ohlc_data['open']),
            np.min(sample_ohlc_data['high']),
            np.min(sample_ohlc_data['low']),
            np.min(sample_ohlc_data['close'])
        )
        max_price = max(
            np.max(sample_ohlc_data['open']),
            np.max(sample_ohlc_data['high']),
            np.max(sample_ohlc_data['low']),
            np.max(sample_ohlc_data['close'])
        )
        assert np.all((result >= min_price) & (result <= max_price))
    
    def test_wave_with_list_input(self):
        """Test wave function with list inputs."""
        open_list = [100.0, 101.0, 102.0, 103.0, 104.0]
        high_list = [105.0, 106.0, 107.0, 108.0, 109.0]
        low_list = [95.0, 96.0, 97.0, 98.0, 99.0]
        close_list = [102.0, 103.0, 104.0, 105.0, 106.0]
        
        result = wave(open=open_list, high=high_list, low=low_list, close=close_list)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1  # Wave function returns a 1D array
        assert len(result) > 0
    
    def test_wave_with_different_length_arrays(self):
        """Test wave function with arrays of different lengths."""
        open_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high_prices = np.array([105.0, 106.0, 107.0, 108.0])  # One element shorter
        low_prices = np.array([95.0, 96.0, 97.0, 98.0, 99.0])
        close_prices = np.array([102.0, 103.0, 104.0, 105.0, 106.0])
        
        # This should raise a ValueError
        with pytest.raises(ValueError, match="All price arrays must have the same length"):
            wave(open=open_prices, high=high_prices, low=low_prices, close=close_prices)
    
    def test_wave_with_numpy_arrays(self):
        """Test wave function with numpy arrays."""
        # Create sample data with specific patterns
        open_prices = np.array([100.0, 102.0, 104.0, 103.0, 105.0])
        high_prices = np.array([105.0, 107.0, 108.0, 106.0, 110.0])
        low_prices = np.array([98.0, 100.0, 102.0, 101.0, 103.0])
        close_prices = np.array([102.0, 104.0, 103.0, 105.0, 107.0])
        
        result = wave(open=open_prices, high=high_prices, low=low_prices, close=close_prices)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1  # Wave function returns a 1D array
        assert len(result) > 0
    
    def test_wave_with_flat_prices(self):
        """Test wave function with flat price data."""
        # Create flat price data
        open_prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        high_prices = np.array([105.0, 105.0, 105.0, 105.0, 105.0])
        low_prices = np.array([95.0, 95.0, 95.0, 95.0, 95.0])
        close_prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        result = wave(open=open_prices, high=high_prices, low=low_prices, close=close_prices)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1  # Wave function returns a 1D array
        assert len(result) > 0
    
    def test_wave_with_trending_prices(self):
        """Test wave function with trending price data."""
        # Create trending price data
        open_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        high_prices = np.array([105.0, 106.0, 107.0, 108.0, 109.0])
        low_prices = np.array([95.0, 96.0, 97.0, 98.0, 99.0])
        close_prices = np.array([101.0, 102.0, 103.0, 104.0, 105.0])
        
        result = wave(open=open_prices, high=high_prices, low=low_prices, close=close_prices)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1  # Wave function returns a 1D array
        assert len(result) > 0


class TestZigzagFunction:
    """Test suite for the zigzag function."""
    
    def test_zigzag_with_1d_array(self, sample_price_data):
        """Test zigzag function with 1D array input."""
        result = zigzag(sample_price_data, threshold=0.03)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # Should be a 2D array with [index, price] columns
        assert result.shape[1] == 2  # Should have 2 columns
        assert len(result) > 0  # Should have some zigzag points
        
        # First point should be the first price
        assert result[0, 1] == sample_price_data[0]
        
        # Indices should be in ascending order
        assert np.all(np.diff(result[:, 0]) > 0)
    
    def test_zigzag_with_list_input(self):
        """Test zigzag function with list input."""
        price_list = [100.0, 105.0, 103.0, 107.0, 109.0, 105.0, 110.0]
        
        result = zigzag(price_list, threshold=0.03)
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2
        assert len(result) > 0
    
    def test_zigzag_with_different_thresholds(self, sample_price_data):
        """Test zigzag function with different threshold values."""
        # Higher threshold should result in fewer points
        result_high = zigzag(sample_price_data, threshold=0.05)
        result_low = zigzag(sample_price_data, threshold=0.01)
        
        assert len(result_high) <= len(result_low)
    
    def test_zigzag_with_empty_array(self):
        """Test zigzag function with an empty array."""
        empty_array = np.array([])
        
        # This should return an empty 2D array
        result = zigzag(empty_array, threshold=0.03)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 2)
    
    def test_zigzag_with_single_value(self):
        """Test zigzag function with a single value."""
        single_value = np.array([100.0])
        
        # This should return the input array directly
        result = zigzag(single_value, threshold=0.03)
        
        # Check the basic properties
        assert isinstance(result, np.ndarray)
        
        # For a single value, the function returns the input array directly
        # So we just check that the value is preserved
        assert 100.0 in result
    
    def test_zigzag_with_flat_data(self):
        """Test zigzag function with flat price data."""
        # Create flat price data
        flat_data = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        result = zigzag(flat_data, threshold=0.03)
        
        # Should have at least the first point
        assert len(result) >= 1
        assert 100.0 in result[:, 1]
        
        # If there's a second point, it should also be 100.0
        if len(result) > 1:
            assert result[-1, 1] == 100.0
    
    def test_zigzag_with_trending_data(self):
        """Test zigzag function with trending price data."""
        # Create trending price data
        trending_data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        
        result = zigzag(trending_data, threshold=0.03)
        
        # Should have at least the first point
        assert len(result) >= 1
        assert 100.0 in result[:, 1]
        
        # If there's a second point, it should be the last value
        if len(result) > 1:
            assert 104.0 in result[:, 1]
    
    def test_zigzag_with_volatile_data(self):
        """Test zigzag function with volatile price data."""
        # Create volatile price data with clear reversals
        volatile_data = np.array([100.0, 110.0, 105.0, 115.0, 105.0, 120.0, 110.0])
        
        result = zigzag(volatile_data, threshold=0.03)
        
        # Should have multiple points due to volatility
        assert len(result) > 2
        
        # First value should be included
        assert 100.0 in result[:, 1]
        
        # Check that zigzag alternates between highs and lows
        if len(result) > 2:
            diffs = np.diff(result[:, 1])
            # At least some differences should be positive and some negative
            assert np.any(diffs > 0) and np.any(diffs < 0)
    
    def test_zigzag_preserves_extremes(self):
        """Test that zigzag preserves local extremes."""
        # Create data with clear local extremes
        data = np.array([100.0, 105.0, 110.0, 105.0, 100.0, 95.0, 100.0, 105.0, 110.0])
        
        result = zigzag(data, threshold=0.05)
        
        # Should include at least some of the extreme points
        max_indices = np.where(data == np.max(data))[0]
        min_indices = np.where(data == np.min(data))[0]
        
        # Extract indices from result
        result_indices = result[:, 0].astype(int)
        
        # Check if at least one max or min index is in the result
        max_found = any(idx in result_indices for idx in max_indices)
        min_found = any(idx in result_indices for idx in min_indices)
        
        assert max_found or min_found


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
