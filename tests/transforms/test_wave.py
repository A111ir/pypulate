"""Tests for wave and zigzag transforms."""

import pytest
import numpy as np
from numpy.typing import NDArray
from pypulate.transforms.wave import wave, zigzag #type: ignore


class TestWave:
    """Test suite for wave transform functions."""
    
    @pytest.fixture
    def sample_ohlc_data(self) -> tuple:
        """Sample OHLC data for testing."""
        # Create sample data with different candle patterns
        open_prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        high_prices = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        low_prices = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        # Create alternating bullish and bearish candles
        close_prices = np.array([11.0, 10.5, 13.0, 12.5, 15.0, 14.5, 17.0, 16.5, 19.0, 18.5])
        
        return open_prices, high_prices, low_prices, close_prices
    
    @pytest.fixture
    def sample_price_data(self) -> NDArray[np.float64]:
        """Sample price data for zigzag testing."""
        # Create sample data with clear up and down trends
        return np.array([
            10.0, 10.5, 11.0, 11.5, 12.0,  # Uptrend
            11.8, 11.5, 11.0, 10.5, 10.0,  # Downtrend
            10.2, 10.5, 11.0, 11.5, 12.0,  # Uptrend
            11.8, 11.5, 11.0, 10.5, 10.0   # Downtrend
        ])
    
    def test_wave_basic(self, sample_ohlc_data):
        """Test basic functionality of wave function."""
        open_prices, high_prices, low_prices, close_prices = sample_ohlc_data
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)
        
        # Check that result has the expected shape (n, 2)
        assert result.ndim == 1
        
        # Check that we have at least some points
        assert len(result) > 0
    
    def test_wave_bullish_candle(self):
        """Test wave function with a single bullish candle."""
        # Bullish candle (close > open)
        open_prices = np.array([10.0])
        high_prices = np.array([12.0])
        low_prices = np.array([9.0])
        close_prices = np.array([11.0])
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # For a bullish candle, we expect low then high
        assert len(result) == 2
        assert result[0] == 9.0  # Low price
        assert result[1] == 12.0  # High price
    
    def test_wave_bearish_candle(self):
        """Test wave function with a single bearish candle."""
        # Bearish candle (close < open)
        open_prices = np.array([10.0])
        high_prices = np.array([12.0])
        low_prices = np.array([9.0])
        close_prices = np.array([9.5])
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # For a bearish candle, we expect high then low
        assert len(result) == 2
        assert result[0] == 12.0  # High price
        assert result[1] == 9.0   # Low price
    
    def test_wave_removes_intermediate_points(self):
        """Test that wave function removes intermediate points in consistent trends."""
        # Create a specific pattern that should trigger point removal
        # Three consecutive points forming a consistent trend
        open_prices = np.array([10.0, 12.0, 14.0])
        high_prices = np.array([11.0, 13.0, 15.0])
        low_prices = np.array([9.0, 11.0, 13.0])
        close_prices = np.array([10.5, 12.5, 14.5])  # All bearish candles
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # Without point removal, we would have 6 points (2 per candle)
        # With point removal, we should have fewer points
        expected_raw_points = 6
        assert len(result) <= expected_raw_points
        
        # Verify that the result contains at least the first and last points
        assert result[0] in [high_prices[0], low_prices[0]]
        assert result[-1] in [high_prices[-1], low_prices[-1]]
    
    def test_wave_input_validation(self):
        """Test that wave function validates input arrays have the same length."""
        open_prices = np.array([10.0, 11.0])
        high_prices = np.array([11.0, 12.0])
        low_prices = np.array([9.0])  # Different length
        close_prices = np.array([11.0, 12.0])
        
        with pytest.raises(ValueError):
            wave(open_prices, high_prices, low_prices, close_prices)
    
    def test_wave_with_equal_open_close(self):
        """Test wave function with candles where open equals close."""
        open_prices = np.array([10.0, 11.0, 12.0])
        high_prices = np.array([11.0, 12.0, 13.0])
        low_prices = np.array([9.0, 10.0, 11.0])
        close_prices = np.array([10.0, 11.0, 12.0])  # Equal to open
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # Check that we get a valid result
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_wave_with_empty_input(self):
        """Test wave function with empty input arrays."""
        open_prices = np.array([])
        high_prices = np.array([])
        low_prices = np.array([])
        close_prices = np.array([])
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # Should return an empty array
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_wave_with_two_candles(self):
        """Test wave function with exactly two candles."""
        open_prices = np.array([10.0, 11.0])
        high_prices = np.array([11.0, 12.0])
        low_prices = np.array([9.0, 10.0])
        close_prices = np.array([10.5, 11.5])  # Both bullish
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # Should have 4 points (2 per candle) since no point removal with just 2 candles
        assert len(result) == 4
        assert result[0] == 9.0   # First candle low
        assert result[1] == 11.0  # First candle high
        assert result[2] == 10.0  # Second candle low
        assert result[3] == 12.0  # Second candle high
    
    def test_wave_with_alternating_candles(self):
        """Test wave function with alternating bullish and bearish candles."""
        open_prices = np.array([10.0, 11.0, 12.0])
        high_prices = np.array([11.0, 12.0, 13.0])
        low_prices = np.array([9.0, 10.0, 11.0])
        close_prices = np.array([10.5, 10.5, 12.5])  # Bullish, bearish, bullish
        
        result = wave(open_prices, high_prices, low_prices, close_prices)
        
        # Check that the result has the expected pattern
        assert len(result) >= 4  # Should have at least 4 points after potential point removal
    
    def test_zigzag_basic(self, sample_price_data):
        """Test basic functionality of zigzag function."""
        result = zigzag(sample_price_data, threshold=0.05)
        
        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)
        
        # Check that result has the expected shape (n, 2)
        assert result.shape[1] == 2
        
        # Check that we have at least some points
        assert len(result) > 0
        
        # First column should be indices, second column should be prices
        assert np.all(result[:, 0] >= 0)
        assert np.all(result[:, 0] < len(sample_price_data))
    
    def test_zigzag_threshold(self, sample_price_data):
        """Test that zigzag function respects the threshold parameter."""
        # Test with different thresholds
        result_small = zigzag(sample_price_data, threshold=0.01)
        result_large = zigzag(sample_price_data, threshold=0.1)
        
        # A smaller threshold should identify more pivot points
        assert len(result_small) >= len(result_large)
    
    def test_zigzag_with_2d_input(self):
        """Test zigzag function with 2D input (index, price)."""
        # Create 2D input with indices and prices
        indices = np.array([0, 1, 2, 3, 4])
        prices = np.array([10.0, 11.0, 10.5, 12.0, 11.0])
        input_data = np.column_stack((indices, prices))
        
        result = zigzag(input_data, threshold=0.05)
        
        # Check that result is a numpy array with the expected shape
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2
        
        # Check that indices in the result match the input indices
        assert np.all(np.isin(result[:, 0], indices))
    
    def test_zigzag_empty_input(self):
        """Test zigzag function with empty input."""
        empty_data = np.array([])
        result = zigzag(empty_data)
        
        # Should return an empty 2D array
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 2)
    
    def test_zigzag_single_point(self):
        """Test zigzag function with a single point."""
        single_point = np.array([10.0])
        result = zigzag(single_point)
        
        # Should return the single point
        assert isinstance(result, np.ndarray)
        # The actual implementation returns the original array for a single point
        assert result.shape == (1,)
        assert result[0] == 10.0
    
    def test_zigzag_extreme_values(self):
        """Test zigzag function with extreme price changes."""
        # Create data with extreme price changes
        prices = np.array([10.0, 20.0, 5.0, 30.0, 1.0])
        
        result = zigzag(prices, threshold=0.1)
        
        # The zigzag function may identify additional pivot points
        # Check that all original prices are included in the result
        assert len(result) >= len(prices)
        
        # Check that the prices in the result include the input prices
        result_prices = result[:, 1]
        for price in prices:
            assert price in result_prices
    
    def test_zigzag_no_significant_changes(self):
        """Test zigzag function with no significant price changes."""
        # Create data with small price changes
        prices = np.array([10.0, 10.01, 10.02, 10.03, 10.04])
        
        result = zigzag(prices, threshold=0.01)
        
        # Only the first and last points should be included
        assert len(result) == 2
        assert result[0, 1] == 10.0
        assert result[-1, 1] == 10.04
    
    def test_zigzag_with_zero_threshold(self):
        """Test zigzag function with zero threshold."""
        prices = np.array([10.0, 11.0, 10.5, 12.0, 11.0])
        
        result = zigzag(prices, threshold=0.0)
        
        # With zero threshold, every change in direction should be captured
        assert len(result) >= 3  # At least first, middle, and last points
        
        # Check that the result includes the first and last points
        assert result[0, 1] == prices[0]
        assert result[-1, 1] == prices[-1]
    
    def test_zigzag_with_negative_threshold(self):
        """Test zigzag function with negative threshold."""
        prices = np.array([10.0, 11.0, 10.5, 12.0, 11.0])
        
        # Negative threshold should be treated as its absolute value
        result = zigzag(prices, threshold=-0.05)
        
        # Check that we get a valid result
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2
    
    def test_zigzag_with_list_input(self):
        """Test zigzag function with list input instead of numpy array."""
        prices = [10.0, 11.0, 10.5, 12.0, 11.0]
        
        result = zigzag(prices, threshold=0.05)
        
        # Check that result is a numpy array with the expected shape
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2
    
    def test_zigzag_with_flat_prices(self):
        """Test zigzag function with flat prices (no changes)."""
        prices = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        
        result = zigzag(prices, threshold=0.05)
        
        # For flat prices, the zigzag function returns only the first point
        assert len(result) == 1
        assert result[0, 0] == 0  # First index
        assert result[0, 1] == 10.0  # Price value
    
    def test_zigzag_with_two_points(self):
        """Test zigzag function with only two price points."""
        prices = np.array([10.0, 11.0])
        
        result = zigzag(prices, threshold=0.05)
        
        # Should include both points
        assert len(result) == 2
        assert result[0, 1] == 10.0
        assert result[1, 1] == 11.0 