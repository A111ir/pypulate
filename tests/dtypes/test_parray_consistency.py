import numpy as np
import pytest
from src.pypulate.dtypes.parray import Parray
import inspect

class TestParrayConsistency:
    """Test consistency between sequential and parallel execution for all methods in the Parray class."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for consistency testing."""
        np.random.seed(42)
        size = 10_000
        # Create a realistic financial time series with trends, cycles, and noise
        t = np.linspace(0, 10, size)
        trend = 0.01 * t
        cycles = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.01 * t)
        noise = 0.1 * np.random.randn(size)
        data = 100 + trend + cycles + noise
        return Parray(data)
    
    @pytest.fixture
    def high_low_close(self):
        """Create high, low, close data for methods that require them."""
        np.random.seed(42)
        size = 10_000
        t = np.linspace(0, 10, size)
        base = 100 + 0.01 * t + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        
        high = base + 0.5 + 0.2 * np.random.randn(size)
        low = base - 0.5 + 0.2 * np.random.randn(size)
        close = base + 0.2 * np.random.randn(size)
        
        return Parray(high), Parray(low), Parray(close)
    
    def check_consistency(self, method_name, seq_array, par_array, *args, **kwargs):
        """Check if a method produces consistent results between sequential and parallel execution."""
        # Get the method by name
        seq_method = getattr(seq_array, method_name)
        par_method = getattr(par_array, method_name)
        
        # Execute sequentially
        seq_result = seq_method(*args, **kwargs)
        
        # Execute in parallel
        par_result = par_method(*args, **kwargs)
        
        # Check if results are the same type
        assert type(seq_result) == type(par_result), f"{method_name}: Result types differ - Sequential: {type(seq_result)}, Parallel: {type(par_result)}"
        
        # Check if results have the same shape (if arrays)
        if isinstance(seq_result, (np.ndarray, Parray)):
            assert seq_result.shape == par_result.shape, f"{method_name}: Result shapes differ - Sequential: {seq_result.shape}, Parallel: {par_result.shape}"
            
            # Check if values are close
            try:
                np.testing.assert_allclose(seq_result, par_result, rtol=1e-5, atol=1e-8, equal_nan=True)
                return True, None
            except AssertionError as e:
                return False, str(e)
        
        # For tuple results (like from bollinger_bands, macd, etc.)
        elif isinstance(seq_result, tuple) and isinstance(par_result, tuple):
            assert len(seq_result) == len(par_result), f"{method_name}: Tuple lengths differ - Sequential: {len(seq_result)}, Parallel: {len(par_result)}"
            
            # Check each element in the tuple
            for i, (seq_item, par_item) in enumerate(zip(seq_result, par_result)):
                if isinstance(seq_item, (np.ndarray, Parray)):
                    assert seq_item.shape == par_item.shape, f"{method_name}[{i}]: Result shapes differ - Sequential: {seq_item.shape}, Parallel: {par_item.shape}"
                    
                    try:
                        np.testing.assert_allclose(seq_item, par_item, rtol=1e-5, atol=1e-8, equal_nan=True)
                    except AssertionError as e:
                        return False, f"Element {i}: {str(e)}"
            
            return True, None
        
        # For dictionary results
        elif isinstance(seq_result, dict) and isinstance(par_result, dict):
            assert seq_result.keys() == par_result.keys(), f"{method_name}: Dictionary keys differ"
            
            # Check each value in the dictionary
            for key in seq_result.keys():
                seq_value = seq_result[key]
                par_value = par_result[key]
                
                if isinstance(seq_value, (np.ndarray, Parray)):
                    assert seq_value.shape == par_value.shape, f"{method_name}[{key}]: Result shapes differ - Sequential: {seq_value.shape}, Parallel: {par_value.shape}"
                    
                    try:
                        np.testing.assert_allclose(seq_value, par_value, rtol=1e-5, atol=1e-8, equal_nan=True)
                    except AssertionError as e:
                        return False, f"Key {key}: {str(e)}"
            
            return True, None
        
        # For other types (like scalars)
        else:
            assert seq_result == par_result, f"{method_name}: Results differ - Sequential: {seq_result}, Parallel: {par_result}"
            return True, None
    
    def test_moving_averages_consistency(self, test_data):
        """Test consistency of moving average methods."""
        methods = [
            ('sma', [9]),
            ('ema', [9]),
            ('wma', [9]),
            ('tma', [9]),
            ('smma', [9]),
            ('zlma', [9]),
            ('hma', [9]),
            ('kama', [9]),
            ('t3', [9, 0.7]),
            ('frama', [9]),
            ('mcginley_dynamic', [9, 0.6])
        ]
        
        inconsistent_methods = []
        
        for method_name, args in methods:
            seq_array = test_data.copy()
            par_array = test_data.copy()
            par_array.enable_parallel()
            
            consistent, error = self.check_consistency(method_name, seq_array, par_array, *args)
            
            if not consistent:
                inconsistent_methods.append((method_name, error))
                print(f"❌ {method_name}: Inconsistent results - {error}")
            else:
                print(f"✅ {method_name}: Consistent results")
        
        assert len(inconsistent_methods) == 0, f"Inconsistent methods: {inconsistent_methods}"
    
    def test_technical_indicators_consistency(self, test_data, high_low_close):
        """Test consistency of technical indicator methods."""
        high, low, close = high_low_close
        
        methods = [
            ('momentum', [test_data], [14]),
            ('roc', [test_data], [14]),
            ('percent_change', [test_data], [1]),
            ('difference', [test_data], [1]),
            ('rsi', [test_data], [14]),
            ('macd', [test_data], [12, 26, 9]),
            ('stochastic_oscillator', [test_data, high, low], [14, 3]),
            ('tsi', [test_data], [25, 13, 7]),
            ('williams_r', [test_data, high, low], [14]),
            ('cci', [test_data], [20]),
            ('adx', [test_data], [14]),
            ('historical_volatility', [test_data], [21]),
            ('atr', [test_data, high, low], [14]),
            ('bollinger_bands', [test_data], [20, 2.0]),
            ('keltner_channels', [test_data, high, low], [20, 10, 2.0]),
            ('donchian_channels', [test_data, high, low], [20]),
            ('volatility_ratio', [test_data], [21, 5])
        ]
        
        inconsistent_methods = []
        
        for method_name, arrays, args in methods:
            seq_arrays = [arr.copy() for arr in arrays]
            par_arrays = [arr.copy() for arr in arrays]
            par_arrays[0].enable_parallel()
            
            consistent, error = self.check_consistency(method_name, seq_arrays[0], par_arrays[0], *([seq_arrays[i] if i > 0 else None for i in range(1, len(seq_arrays))]), *args)
            
            if not consistent:
                inconsistent_methods.append((method_name, error))
                print(f"❌ {method_name}: Inconsistent results - {error}")
            else:
                print(f"✅ {method_name}: Consistent results")
        
        assert len(inconsistent_methods) == 0, f"Inconsistent methods: {inconsistent_methods}"
    
    def test_statistics_consistency(self, test_data):
        """Test consistency of statistical methods."""
        methods = [
            ('descriptive_stats', []),
            ('autocorrelation', [20]),
            ('partial_autocorrelation', [20]),
            ('jarque_bera_test', []),
            ('durbin_watson_test', []),
            ('rolling_max', [14]),
            ('rolling_min', [14]),
            ('rolling_std', [14]),
            ('rolling_var', [14]),
            ('zscore', [14])
        ]
        
        inconsistent_methods = []
        
        for method_name, args in methods:
            seq_array = test_data.copy()
            par_array = test_data.copy()
            par_array.enable_parallel()
            
            consistent, error = self.check_consistency(method_name, seq_array, par_array, *args)
            
            if not consistent:
                inconsistent_methods.append((method_name, error))
                print(f"❌ {method_name}: Inconsistent results - {error}")
            else:
                print(f"✅ {method_name}: Consistent results")
        
        assert len(inconsistent_methods) == 0, f"Inconsistent methods: {inconsistent_methods}"
    
    def test_preprocessing_consistency(self, test_data):
        """Test consistency of preprocessing methods."""
        methods = [
            ('normalize', ['l2']),
            ('standardize', []),
            ('min_max_scale', [(0, 1)]),
            ('robust_scale', ['iqr', (25.0, 75.0)]),
            ('winsorize', [0.05]),
            ('remove_outliers', ['zscore', 3.0]),
            ('log', []),
            ('log_transform', [None, 0.0]),
            ('scale_to_range', [(0.0, 1.0)]),
            ('clip_outliers', [1.0, 99.0])
        ]
        
        inconsistent_methods = []
        
        for method_name, args in methods:
            seq_array = test_data.copy()
            par_array = test_data.copy()
            par_array.enable_parallel()
            
            consistent, error = self.check_consistency(method_name, seq_array, par_array, *args)
            
            if not consistent:
                inconsistent_methods.append((method_name, error))
                print(f"❌ {method_name}: Inconsistent results - {error}")
            else:
                print(f"✅ {method_name}: Consistent results")
        
        assert len(inconsistent_methods) == 0, f"Inconsistent methods: {inconsistent_methods}"
    
    def test_filters_consistency(self, test_data):
        """Test consistency of filter methods."""
        methods = [
            ('kalman_filter', [1e-5, 1e-3, None, 1.0]),
            ('butterworth_filter', [0.1, 4, 'lowpass', 1.0]),
            ('savitzky_golay_filter', [11, 3, 0, 1.0]),
            ('hampel_filter', [5, 3.0]),
            ('hodrick_prescott_filter', [1600.0])
        ]
        
        inconsistent_methods = []
        
        for method_name, args in methods:
            seq_array = test_data.copy()
            par_array = test_data.copy()
            par_array.enable_parallel()
            
            consistent, error = self.check_consistency(method_name, seq_array, par_array, *args)
            
            if not consistent:
                inconsistent_methods.append((method_name, error))
                print(f"❌ {method_name}: Inconsistent results - {error}")
            else:
                print(f"✅ {method_name}: Consistent results")
        
        assert len(inconsistent_methods) == 0, f"Inconsistent methods: {inconsistent_methods}" 