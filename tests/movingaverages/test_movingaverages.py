"""Tests for moving averages module."""

import pytest
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pypulate.moving_averages.movingaverages import (
    sma, ema, wma, tma, smma, zlma, hma, vwma, kama, alma,
    frama, jma, lsma, mcginley_dynamic, t3, vama,
    laguerre_filter, modular_filter, rdma
)


class TestMovingAverages:
    """Test suite for moving averages module."""
    
    @pytest.fixture
    def sample_data(self) -> NDArray[np.float64]:
        """Sample price data for testing."""
        return np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 
                         19.0, 18.0, 17.0, 16.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype=np.float64)
    
    @pytest.fixture
    def sample_volume(self) -> NDArray[np.float64]:
        """Sample volume data for testing."""
        return np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 
                         600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 400.0, 450.0, 500.0, 550.0], 
                         dtype=np.float64)
    
    @pytest.fixture
    def sample_volatility(self) -> NDArray[np.float64]:
        """Sample volatility data for testing."""
        return np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03, 0.02, 
                         0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03, 0.02], 
                         dtype=np.float64)
    
    def test_sma(self, sample_data):
        """Test Simple Moving Average."""
        result = sma(sample_data, period=5)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Check a few values
        assert result[4] == pytest.approx(12.0)  # Average of [10,11,12,13,14]
        assert result[9] == pytest.approx(17.0)  # Average of [15,16,17,18,19]
        assert result[14] == pytest.approx(18.0)  # Average of [20,19,18,17,16]
        
        # Test with different period
        result_period_3 = sma(sample_data, period=3)
        assert np.all(np.isnan(result_period_3[:2]))
        assert result_period_3[2] == pytest.approx(11.0)  # Average of [10,11,12]
        
        # Test with invalid period
        with pytest.raises(ValueError):
            sma(sample_data, period=0)
    
    def test_ema(self, sample_data):
        """Test Exponential Moving Average."""
        result = ema(sample_data, period=5)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with custom alpha
        result_custom_alpha = ema(sample_data, period=5, alpha=0.3)
        assert np.all(np.isnan(result_custom_alpha[:4]))
        
        # Values should be different with different alpha
        assert result[10] != result_custom_alpha[10]
        
        # Test with invalid period
        with pytest.raises(ValueError):
            ema(sample_data, period=0)
    
    def test_wma(self, sample_data):
        """Test Weighted Moving Average."""
        result = wma(sample_data, period=5)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Check a specific value - weighted average calculation
        # For period=5, weights are [1,2,3,4,5]
        # At index 4: (10*1 + 11*2 + 12*3 + 13*4 + 14*5) / (1+2+3+4+5) = 12.666...
        assert result[4] == pytest.approx(12.666666666666666)
        
        # Test with invalid period
        with pytest.raises(ValueError):
            wma(sample_data, period=0)
    
    def test_tma(self, sample_data):
        """Test Triangular Moving Average."""
        result = tma(sample_data, period=5)
        
        # TMA is SMA of SMA, so more values will be NaN at the beginning
        assert np.all(np.isnan(result[:4]))
        
        # Test with invalid period
        with pytest.raises(ValueError):
            tma(sample_data, period=0)
    
    def test_smma(self, sample_data):
        """Test Smoothed Moving Average."""
        result = smma(sample_data, period=5)
        
        # SMMA is equivalent to EMA with alpha=1/period
        ema_result = ema(sample_data, period=5, alpha=1/5)
        np.testing.assert_allclose(result, ema_result)
        
        # Test with invalid period
        with pytest.raises(ValueError):
            smma(sample_data, period=0)
    
    def test_zlma(self, sample_data):
        """Test Zero-Lag Moving Average."""
        result = zlma(sample_data, period=5)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with invalid period
        with pytest.raises(ValueError):
            zlma(sample_data, period=0)
    
    def test_hma(self, sample_data):
        """Test Hull Moving Average."""
        result = hma(sample_data, period=5)
        
        # HMA uses WMA of WMAs, so more values will be NaN at the beginning
        assert np.sum(~np.isnan(result)) > 0  # At least some values should be non-NaN
        
        # Test with invalid period
        with pytest.raises(ValueError):
            hma(sample_data, period=0)
    
    def test_vwma(self, sample_data, sample_volume):
        """Test Volume-Weighted Moving Average."""
        result = vwma(sample_data, sample_volume, period=5)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with mismatched data lengths
        with pytest.raises(ValueError):
            vwma(sample_data, sample_volume[:-1], period=5)
        
        # Test with invalid period
        with pytest.raises(ValueError):
            vwma(sample_data, sample_volume, period=0)
    
    def test_kama(self, sample_data):
        """Test Kaufman Adaptive Moving Average."""
        result = kama(sample_data, period=5, fast_period=2, slow_period=30)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with invalid periods
        with pytest.raises(ValueError):
            kama(sample_data, period=0)
        
        with pytest.raises(ValueError):
            kama(sample_data, period=5, fast_period=0)
        
        with pytest.raises(ValueError):
            kama(sample_data, period=5, fast_period=30, slow_period=2)
    
    def test_alma(self, sample_data):
        """Test Arnaud Legoux Moving Average."""
        result = alma(sample_data, period=5, offset=0.85, sigma=6.0)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            alma(sample_data, period=0)
        
        with pytest.raises(ValueError):
            alma(sample_data, period=5, offset=1.5)
        
        with pytest.raises(ValueError):
            alma(sample_data, period=5, offset=0.5, sigma=0)
    
    def test_frama(self, sample_data):
        """Test Fractal Adaptive Moving Average."""
        result = frama(sample_data, period=5, fc_period=198)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with invalid period
        with pytest.raises(ValueError):
            frama(sample_data, period=0)
        
        with pytest.raises(ValueError):
            frama(sample_data, period=5, fc_period=0)
    
    def test_jma(self, sample_data):
        """Test Jurik Moving Average."""
        result = jma(sample_data, period=5, phase=0)
        
        # First value should be the same as input
        assert result[0] == sample_data[0]
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            jma(sample_data, period=0)
        
        with pytest.raises(ValueError):
            jma(sample_data, period=5, phase=150)
    
    def test_lsma(self, sample_data):
        """Test Least Squares Moving Average."""
        result = lsma(sample_data, period=5)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with invalid period
        with pytest.raises(ValueError):
            lsma(sample_data, period=0)
    
    def test_mcginley_dynamic(self, sample_data):
        """Test McGinley Dynamic Indicator."""
        result = mcginley_dynamic(sample_data, period=5, k=0.6)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with invalid period
        with pytest.raises(ValueError):
            mcginley_dynamic(sample_data, period=0)
    
    def test_t3(self, sample_data):
        """Test Tillson T3 Moving Average."""
        # Create a longer sample to ensure we have non-NaN values
        extended_data = np.tile(sample_data, 3)[:60]
        result = t3(extended_data, period=5, vfactor=0.7)
        
        # T3 uses multiple EMAs, so many values will be NaN at the beginning
        # Check that at least some values are non-NaN
        assert np.sum(~np.isnan(result[30:])) > 0
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            t3(sample_data, period=0)
        
        with pytest.raises(ValueError):
            t3(sample_data, period=5, vfactor=1.5)
    
    def test_vama(self, sample_data, sample_volatility):
        """Test Volatility-Adjusted Moving Average."""
        result = vama(sample_data, sample_volatility, period=5)
        
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        
        # Test with mismatched data lengths
        with pytest.raises(ValueError):
            vama(sample_data, sample_volatility[:-1], period=5)
        
        # Test with invalid period
        with pytest.raises(ValueError):
            vama(sample_data, sample_volatility, period=0)
    
    def test_laguerre_filter(self, sample_data):
        """Test Laguerre Filter."""
        result = laguerre_filter(sample_data, gamma=0.8)
        
        # First value should be NaN
        assert np.isnan(result[0])
        
        # Test with invalid gamma
        with pytest.raises(ValueError):
            laguerre_filter(sample_data, gamma=1.5)
    
    def test_modular_filter(self, sample_data):
        """Test Modular Filter."""
        result = modular_filter(sample_data, period=5, phase=0.5)
        
        # First value should be the same as input
        assert result[0] == sample_data[0]
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            modular_filter(sample_data, period=0)
        
        with pytest.raises(ValueError):
            modular_filter(sample_data, period=5, phase=1.5)
    
    def test_rdma(self, sample_data):
        """Test Rex Dog Moving Average."""
        # RDMA requires at least 200 data points
        extended_data = np.tile(sample_data, 11)[:220]  # Create data with 220 points
        
        result = rdma(extended_data)
        
        # RDMA uses multiple SMAs including SMA200, so first 199 values should be NaN
        assert np.all(np.isnan(result[:199]))
        
        # At least some values should be non-NaN
        assert np.sum(~np.isnan(result)) > 0
    
    def test_input_types(self):
        """Test that functions accept different input types."""
        # Test with list
        data_list = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = sma(data_list, period=3)
        assert isinstance(result, np.ndarray)
        
        # Test with tuple
        data_tuple = (10.0, 11.0, 12.0, 13.0, 14.0)
        result = sma(data_tuple, period=3)
        assert isinstance(result, np.ndarray)
    
    def test_return_types(self, sample_data):
        """Test that functions return NDArray[np.float64]."""
        result = sma(sample_data, period=5)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64 