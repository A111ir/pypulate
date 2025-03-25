"""
Parray Module

This module provides a NumPy array extension that supports method chaining
for financial time series analysis, including moving averages and transforms.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Dict
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ..moving_averages import (
    sma, ema, wma, hma, kama, 
    t3, frama, mcginley_dynamic, tma, smma, zlma
)

from ..technical import (
    momentum, roc, rsi, macd, stochastic_oscillator, 
    tsi, williams_r, cci, percent_change, adx,
    
    bollinger_bands, atr, historical_volatility, 
    keltner_channels, donchian_channels, volatility_ratio,
    
    slope, rolling_max, rolling_min, 
    rolling_std, rolling_var, zscore, log, typical_price
)

from ..transforms.wave import wave, zigzag

from ..filters import (
    kalman_filter, butterworth_filter, savitzky_golay_filter,
    hampel_filter, hodrick_prescott_filter, adaptive_kalman_filter
)

from ..preprocessing.preprocessing import (
    normalize, standardize, winsorize, remove_outliers,
    fill_missing, interpolate_missing, resample, 
    rolling_window, lag_features, difference, log_transform,
    min_max_scale, robust_scale, quantile_transform,
    power_transform, scale_to_range, clip_outliers,
    discretize, polynomial_features, dynamic_tanh
)

from ..preprocessing.statistics import (
    descriptive_stats, correlation_matrix, covariance_matrix,
    autocorrelation, partial_autocorrelation, jarque_bera_test, 
    augmented_dickey_fuller_test, granger_causality_test, ljung_box_test, 
    kpss_test, variance_ratio_test, durbin_watson_test, arch_test,
    kolmogorov_smirnov_test, rolling_statistics, hurst_exponent
)


DEFAULT_NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

DEFAULT_CHUNK_SIZE = 100000  

MAX_CACHE_SIZE = 128

def _process_chunk(chunk_and_args):
    """
    Process a chunk of data with the given function and arguments.
    This function must be at module level to be picklable.
    
    Parameters
    ----------
    chunk_and_args : tuple
        Tuple containing (chunk, func, args, kwargs)
        
    Returns
    -------
    Result of applying func to chunk with args and kwargs
    """
    chunk, func, args, kwargs = chunk_and_args
    return func(chunk, *args, **kwargs)


class Parray(np.ndarray):
    """
    A wrapper around numpy arrays that provides method chaining for financial analysis.
    
    This class allows for fluent method chaining like:
    data.ema(9).sma(20)
    
    It also supports parallel processing and GPU acceleration for improved performance.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.dtypes import Parray
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ts = Parray(data)
    >>> result = ts.ema(3).sma(2)
    >>> # Enable GPU acceleration
    >>> ts.enable_gpu()
    >>> result_gpu = ts.ema(3).sma(2)
    """
    
    def __new__(cls, input_array, memory_optimized=False):
        """
        Create a new Parray instance.
        
        Parameters
        ----------
        input_array : array-like
            Input data to convert to a Parray
        memory_optimized : bool, default=False
            If True, use the smallest possible dtype that can represent the data
            
        Returns
        -------
        Parray
            A new Parray instance
        """
        arr = np.asarray(input_array)
        
        if memory_optimized:
            if np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                min_val, max_val = np.min(arr), np.max(arr)
                
                for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64]:
                    type_info = np.iinfo(dtype)
                    if type_info.min <= min_val and max_val <= type_info.max:
                        arr = arr.astype(dtype)
                        break
            
            elif np.issubdtype(arr.dtype, np.floating):
                if np.allclose(arr, arr.astype(np.float32)):
                    arr = arr.astype(np.float32)
        
        obj = arr.view(cls)
        obj._parallel = False
        obj._num_workers = DEFAULT_NUM_WORKERS
        obj._chunk_size = DEFAULT_CHUNK_SIZE
        obj._gpu = False
        
        return obj
    
    def __array_finalize__(self, obj):
        """
        Finalize the creation of the array.
        
        This method is called whenever a new Parray is created.
        
        Parameters
        ----------
        obj : ndarray
            The array from which the new array was created
        """
        if obj is None: return
        
        self._parallel = getattr(obj, '_parallel', False)
        self._num_workers = getattr(obj, '_num_workers', DEFAULT_NUM_WORKERS)
        self._chunk_size = getattr(obj, '_chunk_size', DEFAULT_CHUNK_SIZE)
        self._gpu = getattr(obj, '_gpu', False)
    
    @property
    def parallel(self):
        """
        Get the parallel processing flag.
        
        Returns
        -------
        bool
            True if parallel processing is enabled, False otherwise
        """
        return self._parallel
    
    @property
    def gpu(self):
        """
        Get the GPU acceleration flag.
        
        Returns
        -------
        bool
            True if GPU acceleration is enabled, False otherwise
        """
        return self._gpu
    
    def enable_gpu(self):
        """
        Enable GPU acceleration for operations that support it.
        
        This requires cupy to be installed. If cupy is not available,
        a warning will be printed and GPU acceleration will not be enabled.
        
        Returns
        -------
        Parray
            Self with GPU acceleration enabled (if available)
        """
        if not CUPY_AVAILABLE:
            import warnings
            warnings.warn(
                "GPU acceleration requested but cupy is not available. "
                "Install cupy to enable GPU acceleration: pip install cupy-cuda11x "
                "(replace 11x with your CUDA version)."
            )
            return self
        
        self._gpu = True
        return self
    
    def disable_gpu(self):
        """
        Disable GPU acceleration.
        
        Returns
        -------
        Parray
            Self with GPU acceleration disabled
        """
        self._gpu = False
        return self
    
    def enable_parallel(self, num_workers=None, chunk_size=None):
        """
        Enable parallel processing for operations that support it.
        
        Parameters
        ----------
        num_workers : int, optional
            Number of worker processes/threads to use.
            If None, use the default number of workers.
        chunk_size : int, optional
            Size of chunks to process in parallel.
            If None, use the default chunk size.
            
        Returns
        -------
        Parray
            Self with parallel processing enabled
        """
        self._parallel = True
        if num_workers is not None:
            self._num_workers = num_workers
        if chunk_size is not None:
            self._chunk_size = chunk_size
        return self
    
    def disable_parallel(self):
        """
        Disable parallel processing.
        
        Returns
        -------
        Parray
            Self with parallel processing disabled
        """
        self._parallel = False
        return self
    
    def optimize_memory(self):
        """
        Optimize memory usage by using the smallest possible dtype.
        
        Returns
        -------
        Parray
            A new Parray with optimized memory usage
        """
        return Parray(self, memory_optimized=True)
    
    def disable_memory_optimization(self):
        """
        Disable memory optimization.
        
        Returns
        -------
        Parray  
        """
        return Parray(self, memory_optimized=False)
    
    def _apply_parallel(self, func, *args, **kwargs):
        """
        Apply a function in parallel chunks.
        
        Parameters
        ----------
        func : callable
            Function to apply
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        Parray
            Result of applying the function in parallel
        """
        if not self._parallel or len(self) < self._chunk_size:
            return func(self, *args, **kwargs)
        
        func_name = getattr(func, '__name__', str(func))
        
        chunk_size = self._chunk_size
        use_thread_pool = True  
        
        
        sequential_funcs = [
            'partial_autocorrelation',
            'kalman_filter', 'butterworth_filter', 'savitzky_golay_filter', 
            'hampel_filter', 'hodrick_prescott_filter', 'adaptive_kalman_filter',
            'granger_causality_test', 'kpss_test', 'ljung_box_test', 'arch_test',
            'variance_ratio_test', 'hurst_exponent',
            'momentum', 'roc', 'percent_change', 'difference'
        ]
        
        window_based_funcs = [
            'sma', 'hma', 'tma', 'smma', 'zlma',  
            'rolling_max', 'rolling_min', 'rolling_std', 'rolling_var',
            'zscore', 'bollinger_bands', 'rsi', 'macd', 'stochastic_oscillator',
            'atr', 'keltner_channels', 'donchian_channels'
        ]
        
        
        elementwise_funcs = [
            '__mul__',  
            '__add__', '__sub__', '__truediv__', '__pow__', 'log', 'exp', 'sqrt',
            'abs', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh'
        ]
        
        preprocessing_funcs = [
            'normalize', 'standardize', 'min_max_scale', 'robust_scale', 'quantile_transform',
            'winsorize', 'remove_outliers', 'fill_missing', 'interpolate_missing', 'log_transform',
            'power_transform', 'scale_to_range', 'clip_outliers', 'discretize', 'polynomial_features'
        ]
        
        if func_name in sequential_funcs:
            return func(self, *args, **kwargs)
        
        elif func_name in window_based_funcs:
            if func_name in ['bollinger_bands', 'keltner_channels', 'donchian_channels']:
                window_size = args[0] if args else kwargs.get('period', 20)
            elif func_name == 'macd':
                fast_period = args[0] if len(args) > 0 else kwargs.get('fast_period', 12)
                slow_period = args[1] if len(args) > 1 else kwargs.get('slow_period', 26)
                signal_period = args[2] if len(args) > 2 else kwargs.get('signal_period', 9)
                window_size = max(fast_period, slow_period, signal_period)
            elif func_name == 'stochastic_oscillator':
                window_size = args[2] if len(args) > 2 else kwargs.get('k_period', 14)
            elif func_name == 'rsi':
                window_size = args[0] if args else kwargs.get('period', 14)
            else:
                window_size = args[0] if args else kwargs.get('period', 20)
            
            chunk_size = max(chunk_size, len(self) // (self._num_workers * 2))
            chunk_size = max(chunk_size, window_size * 3)
            
            if func_name in ['sma', 'hma']:
                chunk_size = max(chunk_size, len(self) // (self._num_workers))
            
            overlap = window_size
            chunks = []
            for i in range(0, len(self), chunk_size):
                start = max(0, i - overlap if i > 0 else 0)
                end = min(len(self), i + chunk_size + overlap)
                chunks.append((self[start:end], start, end, i, i+chunk_size))
            
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                results = list(executor.map(
                    lambda chunk_info: self._process_overlapping_chunk(chunk_info, func, args, kwargs),
                    chunks
                ))
            
            first_result = results[0][0] 
            if isinstance(first_result, tuple):
                num_outputs = len(first_result)
                final_results = []
                
                for output_idx in range(num_outputs):
                    output_dtype = first_result[output_idx].dtype
                    output_array = np.zeros(len(self), dtype=output_dtype)
                    
                    for result_tuple in results:
                        result_component = result_tuple[0][output_idx]
                        start_idx = result_tuple[3] 
                        end_idx = result_tuple[4]  
                        
                        valid_start = 0 if start_idx == 0 else overlap
                        valid_end = len(result_component) if end_idx >= len(self) else len(result_component) - overlap
                        valid_result = result_component[valid_start:valid_end]
                        
                        final_start = start_idx
                        final_end = min(len(self), final_start + len(valid_result))
                        
                        output_array[final_start:final_end] = valid_result
                    
                    final_results.append(Parray(output_array))
                
                return tuple(final_results)
            else:
                final_result = np.zeros(len(self), dtype=first_result.dtype)
                
                for result_tuple in results:
                    result = result_tuple[0] 
                    start_idx = result_tuple[3] 
                    end_idx = result_tuple[4] 
                    
                    valid_start = 0 if start_idx == 0 else overlap
                    valid_end = len(result) if end_idx >= len(self) else len(result) - overlap
                    valid_result = result[valid_start:valid_end]
                    
                    final_start = start_idx
                    final_end = min(len(self), final_start + len(valid_result))
                    
                    final_result[final_start:final_end] = valid_result
                
                return Parray(final_result)
        
        elif func_name in elementwise_funcs:
            if func_name == '__mul__':
                chunk_size = min(chunk_size, len(self) // (self._num_workers * 2))
                chunk_size = max(chunk_size, 50000) 
            else:
                chunk_size = min(chunk_size, len(self) // (self._num_workers * 4))
                chunk_size = max(chunk_size, 10000)  
        
        elif func_name in preprocessing_funcs:
            chunk_size = min(chunk_size, len(self) // (self._num_workers * 2))
            chunk_size = max(chunk_size, 20000) 
        
        chunks = [self[i:i+chunk_size] for i in range(0, len(self), chunk_size)]
        
        chunk_args = [(chunk, func, args, kwargs) for chunk in chunks]
        
        if use_thread_pool:
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                results = list(executor.map(_process_chunk, chunk_args))
        else:
            with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
                results = list(executor.map(_process_chunk, chunk_args))
        
        if all(isinstance(r, np.ndarray) for r in results):
            return np.concatenate(results).view(Parray)
        elif all(isinstance(r, tuple) for r in results) and len(results) > 0:
            transposed = list(zip(*results))
            concatenated = [np.concatenate(arrays).view(Parray) for arrays in transposed]
            return tuple(concatenated)
        else:
            return results
    
    def _process_overlapping_chunk(self, chunk_info, func, args, kwargs):
        """
        Process a chunk with overlapping regions for consistent results at boundaries.
        
        Parameters
        ----------
        chunk_info : tuple
            Tuple containing (chunk_data, start_idx, end_idx, orig_start, orig_end)
        func : callable
            Function to apply
        args, kwargs
            Arguments to pass to the function
            
        Returns
        -------
        tuple
            Tuple containing (result, start_idx, end_idx, orig_start, orig_end)
        """
        chunk, start_idx, end_idx, orig_start, orig_end = chunk_info
        result = func(chunk, *args, **kwargs)
        return result, start_idx, end_idx, orig_start, orig_end
    
    def _apply_gpu(self, func, *args, **kwargs):
        """
        Apply a function using GPU acceleration.
        
        Parameters
        ----------
        func : callable
            Function to apply
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        Parray
            Result of applying the function with GPU acceleration
        """
        if not self._gpu or not CUPY_AVAILABLE:
            return func(self, *args, **kwargs)
        
        cp_array = cp.asarray(self)
        
        try:
            if hasattr(cp, func.__name__):
                cp_func = getattr(cp, func.__name__)
                result = cp_func(cp_array, *args, **kwargs)
            else:
                result = func(cp_array, *args, **kwargs)
            
            if isinstance(result, cp.ndarray):
                return Parray(cp.asnumpy(result))
            elif isinstance(result, tuple) and all(isinstance(r, cp.ndarray) for r in result):
                return tuple(Parray(cp.asnumpy(r)) for r in result)
            else:
                return result
        except Exception as e:
            import warnings
            warnings.warn(
                f"GPU acceleration failed with error: {str(e)}. "
                "Falling back to CPU implementation."
            )
            return func(self, *args, **kwargs)
    
    # -------------------------------------------------------------------------
    # Moving Averages
    # -------------------------------------------------------------------------
    
    def sma(self, period=9) -> 'Parray':
        """Apply Simple Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(sma, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(sma, period)
            return Parray(result)
        return Parray(sma(self, period))
    
    def ema(self, period=9, alpha=None) -> 'Parray':
        """Apply Exponential Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(ema, period, alpha)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(ema, period, alpha)
            return Parray(result)
        return Parray(ema(self, period, alpha))
    
    def wma(self, period=9) -> 'Parray':
        """Apply Weighted Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(wma, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(wma, period)
            return Parray(result)
        return Parray(wma(self, period))
    
    def tma(self, period=9) -> 'Parray':
        """Apply Triangular Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(tma, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(tma, period)
            return Parray(result)
        return Parray(tma(self, period))
    
    def smma(self, period=9) -> 'Parray':
        """Apply Smoothed Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(smma, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(smma, period)
            return Parray(result)
        return Parray(smma(self, period))
    
    def zlma(self, period=9) -> 'Parray':
        """Apply Zero-Lag Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(zlma, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(zlma, period)
            return Parray(result)
        return Parray(zlma(self, period))
    
    def hma(self, period=9) -> 'Parray':
        """Apply Hull Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(hma, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(hma, period)
            return Parray(result)
        return Parray(hma(self, period))
    
    def kama(self, period=9, fast_period=2, slow_period=30) -> 'Parray':
        """Apply Kaufman Adaptive Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(kama, period, fast_period, slow_period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(kama, period, fast_period, slow_period)
            return Parray(result)
        return Parray(kama(self, period, fast_period, slow_period))
    
    def t3(self, period=9, vfactor=0.7):
        """Apply Tillson T3 Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(t3, period, vfactor)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(t3, period, vfactor)
            return Parray(result)
        return Parray(t3(self, period, vfactor))
    
    def frama(self, period=9, fc_period=198) -> 'Parray':
        """Apply Fractal Adaptive Moving Average"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(frama, period, fc_period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(frama, period, fc_period)
            return Parray(result)
        return Parray(frama(self, period, fc_period))
    
    def mcginley_dynamic(self, period=9, k=0.6) -> 'Parray':
        """Apply McGinley Dynamic Indicator"""
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(mcginley_dynamic, period, k)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(mcginley_dynamic, period, k)
            return Parray(result)
        return Parray(mcginley_dynamic(self, period, k))
    
    # -------------------------------------------------------------------------
    # Momentum Indicators
    # -------------------------------------------------------------------------
    
    def momentum(self, period: int = 14) -> 'Parray':
        """
        Calculate momentum over a specified period.
        
        Momentum measures the amount that a price has changed over a given period.
        
        Parameters
        ----------
        period : int, default 14
            Number of periods to calculate momentum
            
        Returns
        -------
        Parray
            Momentum values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(momentum, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(momentum, period)
            return Parray(result)
        return Parray(momentum(self, period))
    
    def roc(self, period: int = 14) -> 'Parray':
        """
        Calculate Rate of Change (ROC) over a specified period.
        
        ROC measures the percentage change in price over a given period.
        
        Parameters
        ----------
        period : int, default 14
            Number of periods to calculate ROC
            
        Returns
        -------
        Parray
            ROC values in percentage
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(roc, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(roc, period)
            return Parray(result)
        return Parray(roc(self, period))
    
    def percent_change(self, periods: int = 1) -> 'Parray':
        """
        Calculate percentage change between consecutive periods.
        
        Parameters
        ----------
        periods : int, default 1
            Number of periods to calculate change over
            
        Returns
        -------
        Parray
            Percentage change values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(percent_change, periods)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(percent_change, periods)
            return Parray(result)
        return Parray(percent_change(self, periods))
    
    def difference(self, periods: int = 1) -> 'Parray':
        """
        Calculate difference between consecutive values.
        
        Parameters
        ----------
        periods : int, default 1
            Number of periods to calculate difference over
            
        Returns
        -------
        Parray
            Difference values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(difference, periods)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(difference, periods)
            return Parray(result)
        return Parray(difference(self, periods))
    
    def rsi(self, period: int = 14, smoothing_type: str = 'sma') -> 'Parray':
        """
        Calculate Relative Strength Index (RSI) over a specified period.
        
        RSI measures the speed and change of price movements, indicating
        overbought (>70) or oversold (<30) conditions.
        
        Parameters
        ----------
        period : int, default 14
            Number of periods to calculate RSI
        smoothing_type : str, default 'sma'
            Type of smoothing to use: 'sma' (Simple Moving Average) or 
            'ema' (Exponential Moving Average)
            
        Returns
        -------
        Parray
            RSI values (0-100)
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(rsi, period, smoothing_type)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(rsi, period, smoothing_type)
            return Parray(result)
        return Parray(rsi(self, period, smoothing_type))
    
    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple['Parray', 'Parray', 'Parray']:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.
        
        Parameters
        ----------
        fast_period : int, default 12
            Period for the fast EMA
        slow_period : int, default 26
            Period for the slow EMA
        signal_period : int, default 9
            Period for the signal line (EMA of MACD line)
            
        Returns
        -------
        tuple of Parray
            Tuple containing (macd_line, signal_line, histogram)
        """
        if self._gpu and CUPY_AVAILABLE:
            macd_line, signal_line, histogram = self._apply_gpu(macd, fast_period, slow_period, signal_period)
            return Parray(macd_line), Parray(signal_line), Parray(histogram)
        elif self._parallel:
            macd_line, signal_line, histogram = self._apply_parallel(macd, fast_period, slow_period, signal_period)
            return Parray(macd_line), Parray(signal_line), Parray(histogram)
        macd_line, signal_line, histogram = macd(self, fast_period, slow_period, signal_period)
        return Parray(macd_line), Parray(signal_line), Parray(histogram)
    
    def stochastic_oscillator(self, high, low, k_period: int = 14, d_period: int = 3) -> Tuple['Parray', 'Parray']:
        """
        Calculate Stochastic Oscillator.
        
        The Stochastic Oscillator is a momentum indicator that shows the location of
        the close relative to the high-low range over a set number of periods.
        
        Parameters
        ----------
        high : numpy.ndarray, optional
            High prices. If None, assumes self contains close prices and high=low=self
        low : numpy.ndarray, optional
            Low prices. If None, assumes self contains close prices and high=low=self
        k_period : int, default 14
            Number of periods for %K
        d_period : int, default 3
            Number of periods for %D (moving average of %K)
            
        Returns
        -------
        tuple of Parray
            Tuple containing (%K, %D)
        """
        if self._gpu and CUPY_AVAILABLE:
            k, d = self._apply_gpu(stochastic_oscillator, high, low, k_period, d_period)
            return Parray(k), Parray(d)
        elif self._parallel:
            k, d = self._apply_parallel(stochastic_oscillator, high, low, k_period, d_period)
            return Parray(k), Parray(d)
        k, d = stochastic_oscillator(self, high, low, k_period, d_period)
        return Parray(k), Parray(d)
    
    def tsi(self, long_period: int = 25, short_period: int = 13, signal_period: int = 7) -> Tuple['Parray', 'Parray']:
        """
        Calculate True Strength Index (TSI).
        
        TSI is a momentum oscillator that helps identify trends and reversals.
        
        Parameters
        ----------
        long_period : int, default 25
            Long period for double smoothing
        short_period : int, default 13
            Short period for double smoothing
        signal_period : int, default 7
            Period for the signal line
            
        Returns
        -------
        tuple of Parray
            Tuple containing (tsi_line, signal_line)
        """
        if self._gpu and CUPY_AVAILABLE:
            tsi_line, signal_line = self._apply_gpu(tsi, long_period, short_period, signal_period)
            return Parray(tsi_line), Parray(signal_line)
        elif self._parallel:
            tsi_line, signal_line = self._apply_parallel(tsi, long_period, short_period, signal_period)
            return Parray(tsi_line), Parray(signal_line)
        tsi_line, signal_line = tsi(self, long_period, short_period, signal_period)
        return Parray(tsi_line), Parray(signal_line)
    
    def williams_r(self, high=None, low=None, period: int = 14) -> 'Parray':
        """
        Calculate Williams %R.
        
        Williams %R is a momentum indicator that measures overbought and oversold levels.
        
        Parameters
        ----------
        high : numpy.ndarray, optional
            High prices. If None, assumes self contains close prices and high=low=self
        low : numpy.ndarray, optional
            Low prices. If None, assumes self contains close prices and high=low=self
        period : int, default 14
            Number of periods for calculation
            
        Returns
        -------
        Parray
            Williams %R values (-100 to 0)
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(williams_r, high, low, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(williams_r, high, low, period)
            return Parray(result)
        result = williams_r(self, high, low, period)
        return Parray(result)
    
    def cci(self, period: int = 20, constant: float = 0.015) -> 'Parray':
        """
        Calculate Commodity Channel Index (CCI).
        
        CCI measures the current price level relative to an average price level over a given period.
        
        Parameters
        ----------
        period : int, default 20
            Number of periods for calculation
        constant : float, default 0.015
            Scaling constant
            
        Returns
        -------
        Parray
            CCI values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(cci, period, constant)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(cci, period, constant)
            return Parray(result)
        result = cci(self, period, constant)
        return Parray(result)
    
    def adx(self, period: int = 14) -> 'Parray':
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures the strength of a trend.
        
        Parameters
        ----------
        period : int, default 14
            Number of periods for calculation
            
        Returns
        -------
        Parray
            ADX values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(adx, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(adx, period)
            return Parray(result)
        result = adx(self, period)
        return Parray(result)
    
    # -------------------------------------------------------------------------
    # Volatility Measurements
    # -------------------------------------------------------------------------
    
    def historical_volatility(self, period: int = 21, annualization_factor: int = 252) -> 'Parray':
        """
        Calculate historical volatility over a specified period.
        
        Historical volatility is the standard deviation of log returns, typically annualized.
        
        Parameters
        ----------
        period : int, default 21
            Number of periods to calculate volatility
        annualization_factor : int, default 252
            Factor to annualize volatility (252 for daily data, 52 for weekly, 12 for monthly)
            
        Returns
        -------
        Parray
            Historical volatility values as percentage
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(historical_volatility, period, annualization_factor)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(historical_volatility, period, annualization_factor)
            return Parray(result)
        result = historical_volatility(self, period, annualization_factor)
        return Parray(result)
    
    def atr(self, high=None, low=None, period: int = 14) -> 'Parray':
        """
        Calculate Average True Range (ATR) over a specified period.
        
        ATR measures market volatility by decomposing the entire range of an asset price.
        
        Parameters
        ----------
        high : numpy.ndarray, optional
            High prices. If None, assumes self contains close prices and high=low=close
        low : numpy.ndarray, optional
            Low prices. If None, assumes self contains close prices and high=low=close
        period : int, default 14
            Number of periods to calculate ATR
            
        Returns
        -------
        Parray
            ATR values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(atr, high, low, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(atr, high, low, period)
            return Parray(result)
        result = atr(self, high, low, period)
        return Parray(result)
    
    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple['Parray', 'Parray', 'Parray']:
        """
        Calculate Bollinger Bands over a specified period.
        
        Bollinger Bands consist of a middle band (SMA), an upper band (SMA + k*std),
        and a lower band (SMA - k*std).
        
        Parameters
        ----------
        period : int, default 20
            Number of periods for the moving average
        std_dev : float, default 2.0
            Number of standard deviations for the upper and lower bands
            
        Returns
        -------
        tuple of Parray
            Tuple containing (upper_band, middle_band, lower_band)
        """
        if self._gpu and CUPY_AVAILABLE:
            upper, middle, lower = self._apply_gpu(bollinger_bands, period, std_dev)
            return Parray(upper), Parray(middle), Parray(lower)
        elif self._parallel:
            upper, middle, lower = self._apply_parallel(bollinger_bands, period, std_dev)
            return Parray(upper), Parray(middle), Parray(lower)
        upper, middle, lower = bollinger_bands(self, period, std_dev)
        return Parray(upper), Parray(middle), Parray(lower)
    
    def keltner_channels(self, high=None, low=None, period: int = 20, atr_period: int = 10, 
                         multiplier: float = 2.0) -> Tuple['Parray', 'Parray', 'Parray']:
        """
        Calculate Keltner Channels over a specified period.
        
        Keltner Channels consist of a middle band (EMA), an upper band (EMA + k*ATR),
        and a lower band (EMA - k*ATR).
        
        Parameters
        ----------
        high : numpy.ndarray, optional
            High prices. If None, assumes self contains close prices and high=low=close
        low : numpy.ndarray, optional
            Low prices. If None, assumes self contains close prices and high=low=close
        period : int, default 20
            Number of periods for the EMA
        atr_period : int, default 10
            Number of periods for the ATR
        multiplier : float, default 2.0
            Multiplier for the ATR
            
        Returns
        -------
        tuple of Parray
            Tuple containing (upper_channel, middle_channel, lower_channel)
        """
        if self._gpu and CUPY_AVAILABLE:
            upper, middle, lower = self._apply_gpu(keltner_channels, high, low, period, atr_period, multiplier)
            return Parray(upper), Parray(middle), Parray(lower)
        elif self._parallel:
            upper, middle, lower = self._apply_parallel(keltner_channels, high, low, period, atr_period, multiplier)
            return Parray(upper), Parray(middle), Parray(lower)
        upper, middle, lower = keltner_channels(self, high, low, period, atr_period, multiplier)
        return Parray(upper), Parray(middle), Parray(lower)
    
    def donchian_channels(self, high=None, low=None, period: int = 20) -> Tuple['Parray', 'Parray', 'Parray']:
        """
        Calculate Donchian Channels over a specified period.
        
        Donchian Channels consist of an upper band (highest high), a lower band (lowest low),
        and a middle band (average of upper and lower).
        
        Parameters
        ----------
        high : numpy.ndarray, optional
            High prices. If None, uses self
        low : numpy.ndarray, optional
            Low prices. If None, uses self
        period : int, default 20
            Number of periods for the channels
            
        Returns
        -------
        tuple of Parray
            Tuple containing (upper_channel, middle_channel, lower_channel)
        """
        if self._gpu and CUPY_AVAILABLE:
            upper, middle, lower = self._apply_gpu(donchian_channels, high, low, period)
            return Parray(upper), Parray(middle), Parray(lower)
        elif self._parallel:
            upper, middle, lower = self._apply_parallel(donchian_channels, high, low, period)
            return Parray(upper), Parray(middle), Parray(lower)
        upper, middle, lower = donchian_channels(self, high, low, period)
        return Parray(upper), Parray(middle), Parray(lower)
    
    def volatility_ratio(self, period: int = 21, smooth_period: int = 5) -> 'Parray':
        """
        Calculate Volatility Ratio over a specified period.
        
        Volatility Ratio compares recent volatility to historical volatility.
        Values above 1 indicate increasing volatility, values below 1 indicate decreasing volatility.
        
        Parameters
        ----------
        period : int, default 21
            Number of periods for historical volatility
        smooth_period : int, default 5
            Number of periods to smooth the ratio
            
        Returns
        -------
        Parray
            Volatility Ratio values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(volatility_ratio, period, smooth_period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(volatility_ratio, period, smooth_period)
            return Parray(result)
        result = volatility_ratio(self, period, smooth_period)
        return Parray(result)
    
    # -------------------------------------------------------------------------
    # Statistical Utility Functions
    # -------------------------------------------------------------------------
    def typical_price(self, high, low) -> 'Parray':
        """
        Calculate the typical price from close, high, and low prices.
        
        Parameters
        ----------
        high : numpy.ndarray, optional
            High prices. If None, uses self
        low : numpy.ndarray, optional
            Low prices. If None, uses self
            
        Returns
        -------
        Parray
            Typical price values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(typical_price, high, low)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(typical_price, high, low)
            return Parray(result)
        result = typical_price(self, high, low)
        return Parray(result)


    def slope(self, period: int = 5) -> 'Parray':
        """
        Calculate the slope of the time series over a specified period.
        
        This method uses linear regression to calculate the slope of the line
        that best fits the data over the specified period.
        
        Parameters
        ----------
        period : int, default 5
            Number of points to use for slope calculation
            
        Returns
        -------
        Parray
            Slope values for each point in the time series
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(slope, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(slope, period)
            return Parray(result)
        result = slope(self, period)
        return Parray(result)
    
    def rolling_max(self, period: int = 14) -> 'Parray':
        """
        Calculate rolling maximum over a specified period.
        
        Parameters
        ----------
        period : int, default 14
            Window size for rolling maximum
            
        Returns
        -------
        Parray
            Rolling maximum values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(rolling_max, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(rolling_max, period)
            return Parray(result)
        result = rolling_max(self, period)
        return Parray(result)
    
    def rolling_min(self, period: int = 14) -> 'Parray':
        """
        Calculate rolling minimum over a specified period.
        
        Parameters
        ----------
        period : int, default 14
            Window size for rolling minimum
            
        Returns
        -------
        Parray
            Rolling minimum values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(rolling_min, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(rolling_min, period)
            return Parray(result)
        result = rolling_min(self, period)
        return Parray(result)
    
    def rolling_std(self, period: int = 14) -> 'Parray':
        """
        Calculate rolling standard deviation over a specified period.
        
        Parameters
        ----------
        period : int, default 14
            Window size for rolling standard deviation
            
        Returns
        -------
        Parray
            Rolling standard deviation values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(rolling_std, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(rolling_std, period)
            return Parray(result)
        result = rolling_std(self, period)
        return Parray(result)
    
    def rolling_var(self, period: int = 14) -> 'Parray':
        """
        Calculate rolling variance over a specified period.
        
        Parameters
        ----------
        period : int, default 14
            Window size for rolling variance
            
        Returns
        -------
        Parray
            Rolling variance values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(rolling_var, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(rolling_var, period)
            return Parray(result)
        result = rolling_var(self, period)
        return Parray(result)

    def zscore(self, period: int = 14) -> 'Parray':
        """
        Calculate rolling z-score over a specified period.
        
        Parameters
        ----------
        period : int, default 14
            Window size for rolling z-score calculation
            
        Returns
        -------
        Parray
            Rolling z-score values
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(zscore, period)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(zscore, period)
            return Parray(result)
        result = zscore(self, period)
        return Parray(result)
    
    def log(self) -> 'Parray':
        """
        Calculate the natural logarithm of the time series.
        
        Returns
        -------
        Parray
            Natural logarithm of the time series
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(log)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(log)
            return Parray(result)
        result = log(self)
        return Parray(result)

    # -------------------------------------------------------------------------
    # Crossover Detection Methods
    # -------------------------------------------------------------------------
    
    def crossover(self, other: Union[np.ndarray, float, int]) -> 'Parray':
        """
        Detect when this series crosses above another series or value.
        
        Parameters
        ----------
        other : array-like or scalar
            The other series or value to compare against
            
        Returns
        -------
        Parray
            Boolean array where True indicates a crossover (this crosses above other)
        
        Examples
        --------
        >>> prices = Parray([10, 11, 12, 11, 10, 9, 10, 11, 12])
        >>> sma = prices.sma(3)
        >>> crossovers = prices.crossover(sma)
        """
        if isinstance(other, (int, float)):
            other = np.full_like(self, other)
            
        other = np.asarray(other)
        
        current_greater = self > other
        prev_less_equal = np.roll(self, 1) <= np.roll(other, 1)
        
        result = np.logical_and(current_greater, prev_less_equal)
        result[0] = False
        
        return Parray(result)
    
    def crossunder(self, other: Union[np.ndarray, float, int]) -> 'Parray':
        """
        Detect when this series crosses below another series or value.
        
        Parameters
        ----------
        other : array-like or scalar
            The other series or value to compare against
            
        Returns
        -------
        Parray
            Boolean array where True indicates a crossunder (this crosses below other)
        
        Examples
        --------
        >>> prices = Parray([10, 11, 12, 11, 10, 9, 10, 11, 12])
        >>> sma = prices.sma(3)
        >>> crossunders = prices.crossunder(sma)
        """
        if isinstance(other, (int, float)):
            other = np.full_like(self, other)
            
        other = np.asarray(other)
        
 
        current_less = self < other
        prev_greater_equal = np.roll(self, 1) >= np.roll(other, 1)
        
        result = np.logical_and(current_less, prev_greater_equal)
        result[0] = False
        
        return Parray(result)
    
    
    # -------------------------------------------------------------------------
    # Transforms
    # -------------------------------------------------------------------------
    
    def wave(self, high=None, low=None, close=None) -> 'Parray':
        """
        Extract wave points from OHLC financial data.
        
        Parameters
        ----------
        high : array-like, optional
            High prices. If None, assumes self contains open prices
        low : array-like, optional
            Low prices. If None, assumes self contains open prices
        close : array-like, optional
            Close prices. If None, assumes self contains open prices
            
        Returns
        -------
        Parray
            Wave points
        """
        if high is None or low is None or close is None:
            raise ValueError("For wave function, you must provide high, low, and close prices")
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(wave, high, low, close)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(wave, high, low, close)
            return Parray(result)
        result = wave(self, high, low, close)
        return Parray(result)
    
    def zigzag(self, threshold: float = 0.03) -> 'Parray':
        """
        Extract zigzag pivot points from price data based on a percentage threshold.
        
        Parameters
        ----------
        threshold : float, default=0.03
            Minimum percentage change required to identify a new pivot point (0.03 = 3%)
            
        Returns
        -------
        Parray
            2D array of zigzag points with shape (n, 2), where each row contains [index, price]
            
        Notes
        -----
        The algorithm identifies significant price movements while filtering out
        minor fluctuations. It marks pivot points where the price changes direction
        by at least the specified threshold percentage.
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(zigzag, threshold)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(zigzag, threshold)
            return Parray(result)
        result = zigzag(self, threshold)
        return Parray(result)
    
    # -------------------------------------------------------------------------
    # Filter Methods
    # -------------------------------------------------------------------------
    
    def kalman_filter(self, process_variance: float = 1e-5, 
                     measurement_variance: float = 1e-3,
                     initial_state: Optional[float] = None,
                     initial_covariance: float = 1.0) -> 'Parray':
        """
        Apply a standard Kalman filter to the time series.
        
        Parameters
        ----------
        process_variance : float, default 1e-5
            Process noise variance (Q)
        measurement_variance : float, default 1e-3
            Measurement noise variance (R)
        initial_state : float, optional
            Initial state estimate. If None, the first data point is used
        initial_covariance : float, default 1.0
            Initial estimate covariance
            
        Returns
        -------
        Parray
            Filtered time series
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(kalman_filter, process_variance, measurement_variance,
                                   initial_state, initial_covariance)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(kalman_filter, process_variance, measurement_variance,
                                   initial_state, initial_covariance)
            return Parray(result)
        result = kalman_filter(self, process_variance, measurement_variance, 
                              initial_state, initial_covariance)
        return Parray(result)
    
    def butterworth_filter(self, cutoff: Union[float, Tuple[float, float]],
                          order: int = 4,
                          filter_type: str = 'lowpass',
                          fs: float = 1.0) -> 'Parray':
        """
        Apply a Butterworth filter to the time series.
        
        Parameters
        ----------
        cutoff : float or tuple of float
            Cutoff frequency. For lowpass and highpass, this is a scalar.
            For bandpass and bandstop, this is a tuple of (low, high)
        order : int, default 4
            Filter order
        filter_type : str, default 'lowpass'
            Filter type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
        fs : float, default 1.0
            Sampling frequency
            
        Returns
        -------
        Parray
            Filtered time series
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(butterworth_filter, cutoff, order, filter_type, fs)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(butterworth_filter, cutoff, order, filter_type, fs)
            return Parray(result)
        result = butterworth_filter(self, cutoff, order, filter_type, fs)
        return Parray(result)
    
    def savitzky_golay_filter(self, window_length: int = 11,
                             polyorder: int = 3,
                             deriv: int = 0,
                             delta: float = 1.0) -> 'Parray':
        """
        Apply a Savitzky-Golay filter to the time series.
        
        Parameters
        ----------
        window_length : int, default 11
            Length of the filter window (must be odd)
        polyorder : int, default 3
            Order of the polynomial used to fit the samples
        deriv : int, default 0
            Order of the derivative to compute
        delta : float, default 1.0
            Spacing of the samples to which the filter is applied
            
        Returns
        -------
        Parray
            Filtered time series
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(savitzky_golay_filter, window_length, polyorder, deriv, delta)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(savitzky_golay_filter, window_length, polyorder, deriv, delta)
            return Parray(result)
        result = savitzky_golay_filter(self, window_length, polyorder, deriv, delta)
        return Parray(result)

    
    def hampel_filter(self, window_size: int = 5, n_sigmas: float = 3.0) -> 'Parray':
        """
        Apply a Hampel filter to the time series to remove outliers.
        
        Parameters
        ----------
        window_size : int, default 5
            Size of the window (number of points on each side of the current point)
        n_sigmas : float, default 3.0
            Number of standard deviations to use for outlier detection
            
        Returns
        -------
        Parray
            Filtered time series
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(hampel_filter, self, window_size, n_sigmas)
            return Parray(result)
        elif self._parallel:
            result = hampel_filter(self, window_size, n_sigmas)
            return Parray(result)
        result = hampel_filter(self, window_size, n_sigmas)
        return Parray(result)
    
    def hodrick_prescott_filter(self, lambda_param: float = 1600.0) -> Tuple['Parray', 'Parray']:
        """
        Apply the Hodrick-Prescott filter to decompose the time series into trend and cycle components.
        
        Parameters
        ----------
        lambda_param : float, default 1600.0
            Smoothing parameter. The larger the value, the smoother the trend component
            
        Returns
        -------
        tuple of Parray
            Tuple containing (trend, cycle) components
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(hodrick_prescott_filter, lambda_param)
            if isinstance(result, tuple) and len(result) == 2:
                return Parray(result[0]), Parray(result[1])  # type: ignore
        trend, cycle = hodrick_prescott_filter(self, lambda_param)
        return Parray(trend), Parray(cycle)
    
    def adaptive_kalman_filter(self, process_variance_init: float = 1e-5,
                              measurement_variance_init: float = 1e-3,
                              adaptation_rate: float = 0.01,
                              window_size: int = 10,
                              initial_state: Optional[float] = None,
                              initial_covariance: float = 1.0) -> 'Parray':
        """
        Apply an adaptive Kalman filter to the time series.
        
        Parameters
        ----------
        process_variance_init : float, default 1e-5
            Initial process noise variance (Q)
        measurement_variance_init : float, default 1e-3
            Initial measurement noise variance (R)
        adaptation_rate : float, default 0.01
            Rate at which the filter adapts to changes
        window_size : int, default 10
            Size of the window for innovation estimation
        initial_state : float, optional
            Initial state estimate. If None, the first data point is used
        initial_covariance : float, default 1.0
            Initial estimate covariance
            
        Returns
        -------
        Parray
            Filtered time series
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(adaptive_kalman_filter, self, process_variance_init, measurement_variance_init,
                                   adaptation_rate, window_size, initial_state, initial_covariance)
            return Parray(result)
        elif self._parallel:
            result = adaptive_kalman_filter(self, process_variance_init, measurement_variance_init,
                                       adaptation_rate, window_size, initial_state, initial_covariance)
            return Parray(result)
        result = adaptive_kalman_filter(self, process_variance_init, measurement_variance_init,
                                       adaptation_rate, window_size, initial_state, initial_covariance)
        return Parray(result)
    
    # -------------------------------------------------------------------------
    # Data Processing Methods
    # -------------------------------------------------------------------------
    
    def normalize(self, method: str = 'l2') -> 'Parray':
        """
        Normalize data using vector normalization methods.
        
        Parameters
        ----------
        method : str, default='l2'
            Normalization method: 'l1' or 'l2'
            - 'l1': L1 normalization (Manhattan norm)
            - 'l2': L2 normalization (Euclidean norm)
        
        Returns
        -------
        Parray
            Normalized data with unit norm
            
        Notes
        -----
        For min-max scaling, use the min_max_scale method instead.
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(normalize, method)
            return Parray(result)
        elif self._parallel:
            result = normalize(self, method)
            return Parray(result)
        result = normalize(self, method)
        return Parray(result)
    
    def normalize_l1(self) -> 'Parray':
        """
        Normalize data using L1 norm (Manhattan norm).
        
        Returns
        -------
        Parray
            Normalized data with unit L1 norm
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(normalize, 'l1')
            return Parray(result)
        elif self._parallel:
            result = normalize(self, 'l1')
            return Parray(result)
        result = normalize(self, 'l1')
        return Parray(result)
    
    def normalize_l2(self) -> 'Parray':
        """
        Normalize data using L2 norm (Euclidean norm).
        
        Returns
        -------
        Parray
            Normalized data with unit L2 norm
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(normalize, 'l2')
            return Parray(result)
        elif self._parallel:
            result = normalize(self, 'l2')
            return Parray(result)
        result = normalize(self, 'l2')
        return Parray(result)
    
    def standardize(self) -> 'Parray':
        """
        Standardize data to have mean 0 and standard deviation 1 (Z-score normalization).
        
        Returns
        -------
        Parray
            Standardized data with zero mean and unit variance
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(standardize)
            return Parray(result)
        elif self._parallel:
            result = standardize(self)
            return Parray(result)
        result = standardize(self)
        return Parray(result)
    
    def min_max_scale(self, feature_range: Tuple[float, float] = (0, 1)) -> 'Parray':
        """
        Scale data to a specified range.
        
        Parameters
        ----------
        feature_range : tuple, default=(0, 1)
            Desired range of the scaled data
            
        Returns
        -------
        Parray
            Scaled data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(min_max_scale, feature_range)
            return Parray(result)
        elif self._parallel:
            result = min_max_scale(self, feature_range)
            return Parray(result)
        result = min_max_scale(self, feature_range)
        return Parray(result)
    
    def robust_scale(self, method: str = 'iqr', 
                quantile_range: Tuple[float, float] = (25.0, 75.0)) -> 'Parray':
        """
        Scale data using robust statistics.
        
        Parameters
        ----------
        method : str, default='iqr'
            Method to use: 'iqr' or 'mad'
        quantile_range : tuple, default=(25.0, 75.0)
            Quantile range to use for IQR method
            
        Returns
        -------
        Parray
            Robustly scaled data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(robust_scale, method, quantile_range)
            return Parray(result)
        elif self._parallel:
            result = robust_scale(self, method, quantile_range)
            return Parray(result)
        result = robust_scale(self, method, quantile_range)
        return Parray(result)
    
    def quantile_transform(self, n_quantiles: int = 1000, 
                          output_distribution: str = 'uniform') -> 'Parray':
        """
        Transform data to follow a uniform or normal distribution.
        
        Parameters
        ----------
        n_quantiles : int, default=1000
            Number of quantiles to use
        output_distribution : str, default='uniform'
            Distribution to transform to: 'uniform' or 'normal'
            
        Returns
        -------
        Parray
            Transformed data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(quantile_transform, n_quantiles, output_distribution)
            return Parray(result)
        elif self._parallel:
            result = quantile_transform(self, n_quantiles, output_distribution)
            return Parray(result)
        result = quantile_transform(self, n_quantiles, output_distribution)
        return Parray(result)
    
    def winsorize(self, limits: Union[float, Tuple[float, float]] = 0.05) -> 'Parray':
        """
        Limit extreme values in data.
        
        Parameters
        ----------
        limits : float or tuple, default=0.05
            Proportion of values to limit
            
        Returns
        -------
        Parray
            Winsorized data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(winsorize, limits)
            return Parray(result)
        elif self._parallel:
            result = winsorize(self, limits)
            return Parray(result)
        result = winsorize(self, limits)
        return Parray(result)
    
    def remove_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> 'Parray':
        """
        Remove outliers from data.
        
        Parameters
        ----------
        method : str, default='zscore'
            Method to use: 'zscore', 'iqr', or 'mad'
        threshold : float, default=3.0
            Threshold for outlier detection
            
        Returns
        -------
        Parray
            Data with outliers removed
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(remove_outliers, method, threshold)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(remove_outliers, method, threshold)
            return Parray(result)
        result = remove_outliers(self, method, threshold)
        return Parray(result)
    
    def fill_missing(self, method: str = 'mean', value: Optional[float] = None) -> 'Parray':
        """
        Fill missing values in data.
        
        Parameters
        ----------
        method : str, default='mean'
            Method to fill missing values: 'mean', 'median', 'mode', 'forward', 'backward', 'value'
        value : float, optional
            Value to use when method='value'
            
        Returns
        -------
        Parray
            Data with missing values filled
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(fill_missing, method, value)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(fill_missing, method, value)
            return Parray(result)
        result = fill_missing(self, method, value)
        return Parray(result)
    
    def interpolate_missing(self, method: str = 'linear') -> 'Parray':
        """
        Interpolate missing values in data.
        
        Parameters
        ----------
        method : str, default='linear'
            Interpolation method: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            
        Returns
        -------
        Parray
            Data with missing values interpolated
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(interpolate_missing, method)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(interpolate_missing, method)
            return Parray(result)
        result = interpolate_missing(self, method)
        return Parray(result)
    
    
    def log_transform(self, base: Optional[float] = None, offset: float = 0.0) -> 'Parray':
        """
        Apply logarithmic transformation.
        
        Parameters
        ----------
        base : float, optional
            Base of the logarithm. If None, natural logarithm is used.
        offset : float, default=0.0
            Offset to add to data before taking logarithm
            
        Returns
        -------
        Parray
            Log-transformed data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(log_transform, base, offset)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(log_transform, base, offset)
            return Parray(result)
        result = log_transform(self, base, offset)
        return Parray(result)
    
    def power_transform(self, method: str = 'yeo-johnson', standardize: bool = True) -> 'Parray':
        """
        Apply power transformation to make data more Gaussian-like.
        
        Parameters
        ----------
        method : str, default='yeo-johnson'
            The power transform method: 'box-cox' or 'yeo-johnson'
        standardize : bool, default=True
            Whether to standardize the data after transformation
            
        Returns
        -------
        Parray
            Power transformed data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(power_transform, method, standardize)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(power_transform, method, standardize)
            return Parray(result)
        result = power_transform(self, method, standardize)
        return Parray(result)
    
    def scale_to_range(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> 'Parray':
        """
        Scale data to a specified range.
        
        Parameters
        ----------
        feature_range : tuple, default=(0.0, 1.0)
            Desired range of the scaled data
            
        Returns
        -------
        Parray
            Scaled data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(scale_to_range, feature_range)
            return Parray(result)
        elif self._parallel:
            result = scale_to_range(self, feature_range)
            return Parray(result)
        result = scale_to_range(self, feature_range)
        return Parray(result)
    
    def clip_outliers(self, lower_percentile: float = 1.0, 
                 upper_percentile: float = 99.0) -> 'Parray':
        """
        Clip values outside specified percentiles.
        
        Parameters
        ----------
        lower_percentile : float, default=1.0
            Lower percentile to clip
        upper_percentile : float, default=99.0
            Upper percentile to clip
            
        Returns
        -------
        Parray
            Clipped data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(clip_outliers, lower_percentile, upper_percentile)
            return Parray(result)
        elif self._parallel:
            result = clip_outliers(self, lower_percentile, upper_percentile)
            return Parray(result)
        result = clip_outliers(self, lower_percentile, upper_percentile)
        return Parray(result)
    
    def discretize(self, n_bins: int = 5, strategy: str = 'uniform') -> 'Parray':
        """
        Discretize continuous data into bins.
        
        Parameters
        ----------
        n_bins : int, default=5
            Number of bins to create
        strategy : str, default='uniform'
            Strategy to use for creating bins: 'uniform', 'quantile', or 'kmeans'
            
        Returns
        -------
        Parray
            Array of bin labels (1 to n_bins)
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(discretize, n_bins, strategy)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(discretize, n_bins, strategy)
            return Parray(result)
        result = discretize(self, n_bins, strategy)
        return Parray(result)
    
    def polynomial_features(self, degree: int = 2) -> 'Parray':
        """
        Generate polynomial features up to specified degree.
        
        Parameters
        ----------
        degree : int, default=2
            Maximum degree of polynomial features
            
        Returns
        -------
        Parray
            Array with polynomial features as columns
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(polynomial_features, degree)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(polynomial_features, degree)
            return Parray(result)
        result = polynomial_features(self, degree)
        return Parray(result)
    
    def resample(self, factor: int, method: str = 'mean') -> 'Parray':
        """
        Resample data by aggregating values.
        
        Parameters
        ----------
        factor : int
            Resampling factor (e.g., 5 means aggregate every 5 points)
        method : str, default='mean'
            Aggregation method: 'mean', 'median', 'sum', 'min', 'max'
            
        Returns
        -------
        Parray
            Resampled data
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(resample, factor, method)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(resample, factor, method)
            return Parray(result)
        result = resample(self, factor, method)
        return Parray(result)
    
    def rolling_window(self, window_size: int, step: int = 1) -> 'Parray':
        """
        Create rolling windows from data.
        
        Parameters
        ----------
        window_size : int
            Size of each window
        step : int, default=1
            Step size between windows
            
        Returns
        -------
        Parray
            Array with rolling windows as rows
        """
        
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(rolling_window, window_size, step)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(rolling_window, window_size, step)
            return Parray(result)
        result = rolling_window(self, window_size, step)
        return Parray(result)
    
    def dynamic_tanh(self, alpha: float = 1.0) -> 'Parray':
        """
        Apply Dynamic Tanh (DyT) transformation to data, which helps normalize data 
        while preserving relative differences and handling outliers well.
        
        Parameters
        ----------
        alpha : float, default=1.0
            Scaling factor that controls the transformation intensity.
            Higher values lead to more aggressive normalization (less extreme values).
            
        Returns
        -------
        Parray
            DyT-transformed data with values in range (-1, 1)
            
        Notes
        -----
        The Dynamic Tanh (DyT) transformation follows these steps:
        1. Center data by subtracting the median
        2. Scale data by dividing by (MAD * alpha), where MAD is Median Absolute Deviation
           Higher alpha means more scaling (division by larger value) before tanh
        3. Apply tanh transformation to the scaled data
        
        This transformation is particularly useful for financial data as it:
        - Is robust to outliers (uses median and MAD instead of mean and std)
        - Maps all values to the range (-1, 1) without clipping extreme values
        - Preserves the shape of the distribution better than min-max scaling
        - Handles multi-modal distributions better than standard normalization
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(dynamic_tanh, alpha)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(dynamic_tanh, alpha)
            return Parray(result)
        result = dynamic_tanh(self, alpha)
        return Parray(result)
    
    # -------------------------------------------------------------------------
    # Statistics Methods
    # -------------------------------------------------------------------------

    def rolling_statistics(self, window: int, statistics: List[str]) -> Dict[str, 'Parray']:
        """
        Calculate multiple rolling statistics in a single pass.
        
        Parameters
        ----------
        window : int
            Size of the rolling window
        statistics : list of str
            List of statistics to calculate. Options include:
            'mean', 'std', 'min', 'max', 'median', 'skew', 'kurt'
            
        Returns
        -------
        dict
            Dictionary with keys as statistic names and values as Parray objects
        """
        from ..preprocessing.statistics import rolling_statistics
        
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(rolling_statistics, window, statistics)
            return {stat: Parray(result[stat]) for stat in statistics}
        elif self._parallel:
            result = self._apply_parallel(rolling_statistics, window, statistics)
            return {stat: Parray(result[stat]) for stat in statistics}
        result = rolling_statistics(self, window, statistics)
        return {stat: Parray(result[stat]) for stat in statistics}
    
    def descriptive_stats(self) -> Dict[str, float]:
        """
        Calculate descriptive statistics for data.
        
        Returns
        -------
        dict
            Dictionary of statistics including mean, median, std, min, max, etc.
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(descriptive_stats)
            if isinstance(gpu_result, dict):
                return gpu_result
        elif self._parallel:
            result = descriptive_stats(self)
            return result
        result = descriptive_stats(self)
        return result
    
    def autocorrelation(self, max_lag: int = 20) -> 'Parray':
        """
        Calculate autocorrelation function for time series data.
        
        Parameters
        ----------
        max_lag : int, default=20
            Maximum lag to calculate autocorrelation
            
        Returns
        -------
        Parray
            Autocorrelation values for each lag
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(autocorrelation, max_lag)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(autocorrelation, max_lag)
            return Parray(result)
        result = autocorrelation(self, max_lag)
        return Parray(result)
    
    def partial_autocorrelation(self, max_lag: int = 20) -> 'Parray':
        """
        Calculate partial autocorrelation function for time series data.
        
        Parameters
        ----------
        max_lag : int, default=20
            Maximum lag to calculate partial autocorrelation
            
        Returns
        -------
        Parray
            Partial autocorrelation values for each lag
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(partial_autocorrelation, max_lag)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(partial_autocorrelation, max_lag)
            return Parray(result)
        result = partial_autocorrelation(self, max_lag)
        return Parray(result)
    
    def jarque_bera_test(self) -> Tuple[float, float]:
        """
        Perform Jarque-Bera test for normality.
        
        Returns
        -------
        tuple
            (test statistic, p-value)
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(jarque_bera_test, self)
            if isinstance(gpu_result, tuple):
                return gpu_result
        elif self._parallel:
            result = jarque_bera_test(self)
            return result
        result = jarque_bera_test(self)
        return result
    
    def augmented_dickey_fuller_test(self, regression: str = 'c', 
                               max_lag: Optional[int] = None) -> Tuple[float, float]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Parameters
        ----------
        regression : str, default='c'
            Regression model to use: 'c' for constant, 'ct' for constant and trend,
            'ctt' for constant, trend, and trend squared, 'n' for no regression
        max_lag : int, optional
            Maximum lag to consider
            
        Returns
        -------
        tuple
            (test statistic, p-value)
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(augmented_dickey_fuller_test, regression, max_lag)
            if isinstance(gpu_result, tuple):
                return gpu_result
        elif self._parallel:
            result = augmented_dickey_fuller_test(self, regression, max_lag)
            return result
        result = augmented_dickey_fuller_test(self, regression, max_lag)
        return result
    
    def granger_causality_test(self, y: Union[np.ndarray, 'Parray'], max_lag: int = 1) -> Tuple[float, float]:
        """
        Perform Granger causality test.
        
        Parameters
        ----------
        y : array-like
            Second time series
        max_lag : int, default=1
            Maximum lag to consider
            
        Returns
        -------
        tuple
            (test statistic, p-value)
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(granger_causality_test, y, max_lag)
            if isinstance(gpu_result, tuple):
                return gpu_result
        elif self._parallel:
            result = granger_causality_test(self, y, max_lag)
            return result
        result = granger_causality_test(self, y, max_lag)
        return result
    
    def kpss_test(self, regression: str = 'c', lags: Optional[int] = None) -> Tuple[float, float]:
        """
        Perform KPSS test for stationarity.
        
        Parameters
        ----------
        regression : str, default='c'
            Regression model to use: 'c' for constant, 'ct' for constant and trend
        lags : int, optional
            Number of lags to use
            
        Returns
        -------
        tuple
            (test statistic, p-value)
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(kpss_test, regression, lags)
            if isinstance(gpu_result, tuple):
                return gpu_result
        elif self._parallel:
            result = kpss_test(self, regression, lags)
            return result
        result = kpss_test(self, regression, lags)
        return result
    
    def ljung_box_test(self, lags: int = 10, boxpierce: bool = False) -> Union[Tuple[float, float], Dict[int, Tuple[float, float]]]:
        """
        Perform Ljung-Box test for autocorrelation.
        
        Parameters
        ----------
        lags : int, default=10
            Number of lags to use
        boxpierce : bool, default=False
            If True, use Box-Pierce test instead of Ljung-Box
            
        Returns
        -------
        tuple or dict
            (test statistic, p-value) or {lag: (test statistic, p-value)}
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(ljung_box_test, lags, boxpierce)
            if isinstance(gpu_result, tuple) or isinstance(gpu_result, dict):
                return gpu_result 
        elif self._parallel:
            result = ljung_box_test(self, lags, boxpierce)
            return result 
        result = ljung_box_test(self, lags, boxpierce)
        return result 
    
    def durbin_watson_test(self) -> float:
        """
        Perform Durbin-Watson test for autocorrelation.
        
        Returns
        -------
        float
            Test statistic
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(durbin_watson_test)
            if isinstance(gpu_result, float):
                return gpu_result 
        elif self._parallel:
            result = durbin_watson_test(self)
            return result 
        result = durbin_watson_test(self)
        return result 
    
    def arch_test(self, lags: int = 5) -> Tuple[float, float]:
        """
        Perform ARCH test for heteroskedasticity.
        
        Parameters
        ----------
        lags : int, default=5
            Number of lags to use
            
        Returns
        -------
        tuple
            (test statistic, p-value)
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(arch_test, lags)
            if isinstance(gpu_result, tuple):
                return gpu_result
        elif self._parallel:
            result = arch_test(self, lags)
            return result
        result = arch_test(self, lags)
        return result
    
    def kolmogorov_smirnov_test(self, dist: str = 'norm',
                           params: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution fit.
        
        Parameters
        ----------
        dist : str, default='norm'
            Distribution to test against
        params : dict, optional
            Parameters for the distribution
            
        Returns
        -------
        tuple
            (test statistic, p-value)
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(kolmogorov_smirnov_test, dist, params)
            if isinstance(gpu_result, tuple):
                return gpu_result
        elif self._parallel:
            result = kolmogorov_smirnov_test(self, dist, params)
            return result
        result = kolmogorov_smirnov_test(self, dist, params)
        return result
    
    def hurst_exponent(self, max_lag: Optional[int] = None) -> float:
        """
        Calculate Hurst exponent.
        
        Parameters
        ----------
        max_lag : int, optional
            Maximum lag to consider
            
        Returns
        -------
        float
            Hurst exponent
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(hurst_exponent, max_lag)
            if isinstance(gpu_result, float):
                return gpu_result 
        elif self._parallel:
            result = hurst_exponent(self, max_lag)
            return result 
        result = hurst_exponent(self, max_lag)
        return result 
    
    def variance_ratio_test(self, periods: Optional[List[int]] = None,
                       robust: bool = True) -> Dict[int, Tuple[float, float]]:
        """
        Perform variance ratio test.
        
        Parameters
        ----------
        periods : list of int, optional
            Periods to test
        robust : bool, default=True
            Whether to use robust standard errors
            
        Returns
        -------
        dict
            {period: (test statistic, p-value)}
        """
        if self._gpu and CUPY_AVAILABLE:
            gpu_result = self._apply_gpu(variance_ratio_test, periods, robust)
            if isinstance(gpu_result, dict):
                return gpu_result
        elif self._parallel:
            result = variance_ratio_test(self, periods, robust)
            return result
        result = variance_ratio_test(self, periods, robust)
        return result
    
    def correlation_matrix(self, method: str = 'pearson', 
                          min_periods: int = 1) -> 'Parray':
        """
        Calculate correlation matrix for multivariate data.
        
        Parameters
        ----------
        method : str, default='pearson'
            Correlation method: 'pearson', 'spearman', or 'kendall'
        min_periods : int, default=1
            Minimum number of observations required per pair of columns
            
        Returns
        -------
        Parray
            Correlation matrix
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(correlation_matrix, method, min_periods)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(correlation_matrix, method, min_periods)
            return Parray(result)
        result = correlation_matrix(self, method, min_periods)
        return Parray(result)
    
    def covariance_matrix(self, ddof: int = 1) -> 'Parray':
        """
        Calculate covariance matrix for multivariate data.
        
        Parameters
        ----------
        ddof : int, default=1
            Delta degrees of freedom
            
        Returns
        -------
        Parray
            Covariance matrix
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(covariance_matrix, ddof)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(covariance_matrix, ddof)
            return Parray(result)
        result = covariance_matrix(self, ddof)
        return Parray(result)
    
    def lag_features(self, lags: List[int]) -> 'Parray':
        """
        Create lagged features from data.
        
        Parameters
        ----------
        lags : list of int
            List of lag values. Zero lag returns the original values,
            negative lags are ignored, and lags larger than the data length
            result in NaN columns.
            
        Returns
        -------
        Parray
            Array with original and lagged features as columns
        """
        if self._gpu and CUPY_AVAILABLE:
            result = self._apply_gpu(lag_features, lags)
            return Parray(result)
        elif self._parallel:
            result = self._apply_parallel(lag_features, lags)
            return Parray(result)
        result = lag_features(self, lags)
        return Parray(result)
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _memoize(func):
        """
        Decorator to memoize a function.
        
        Parameters
        ----------
        func : callable
            Function to memoize
            
        Returns
        -------
        callable
            Memoized function
        """
        @functools.lru_cache(maxsize=MAX_CACHE_SIZE)
        def memoized_func(*args, **kwargs):
            return func(*args, **kwargs)
        return memoized_func
    
    @staticmethod
    def is_gpu_available():
        """
        Check if GPU acceleration is available.
        
        Returns
        -------
        bool
            True if GPU acceleration is available, False otherwise
        """
        return CUPY_AVAILABLE
    
    @staticmethod
    def get_gpu_info():
        """
        Get information about available GPUs.
        
        Returns
        -------
        dict or None
            Dictionary containing GPU information if available, None otherwise
        """
        if not CUPY_AVAILABLE:
            return None
        
        try:
            # Get number of devices
            num_devices = cp.cuda.runtime.getDeviceCount()
            
            # Get information for each device
            devices = []
            for i in range(num_devices):
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                devices.append({
                    'id': i,
                    'name': device_props['name'].decode('utf-8'),
                    'total_memory': device_props['totalGlobalMem'],
                    'compute_capability': f"{device_props['major']}.{device_props['minor']}",
                    'multi_processor_count': device_props['multiProcessorCount']
                })
            
            return {
                'num_devices': num_devices,
                'devices': devices,
                'cupy_version': cp.__version__
            }
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to get GPU information: {str(e)}")
            return None
    
    @classmethod
    def from_chunks(cls, chunks, axis=0):
        """
        Create a Parray from chunks of data.
        
        This is useful for processing large datasets that don't fit in memory.
        
        Parameters
        ----------
        chunks : list of array-like
            Chunks of data to combine
        axis : int, default=0
            Axis along which to concatenate the chunks
            
        Returns
        -------
        Parray
            A new Parray created from the chunks
        """
        return cls(np.concatenate([np.asarray(chunk) for chunk in chunks], axis=axis))
    
    def to_chunks(self, chunk_size=None):
        """
        Split the array into chunks.
        
        Parameters
        ----------
        chunk_size : int, optional
            Size of each chunk. If None, use the default chunk size.
            
        Returns
        -------
        list of Parray
            List of chunks
        """
        if chunk_size is None:
            chunk_size = self._chunk_size
        
        return [Parray(self[i:i+chunk_size]) for i in range(0, len(self), chunk_size)]
    
    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the array.
        
        Parameters
        ----------
        func : callable
            Function to apply
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        Parray
            Result of applying the function
        """
        if self._parallel:
            return self._apply_parallel(func, *args, **kwargs)
        
        result = func(self, *args, **kwargs)
        if isinstance(result, np.ndarray):
            return Parray(result)
        return result
    
    def apply_along_axis(self, func, axis, *args, **kwargs):
        """
        Apply a function along a specified axis.
        
        Parameters
        ----------
        func : callable
            Function to apply
        axis : int
            Axis along which to apply the function
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        Parray
            Result of applying the function along the axis
        """
        result = np.apply_along_axis(func, axis, self, *args, **kwargs)
        return Parray(result)
    
    def rolling_apply(self, window, func, *args, **kwargs):
        """
        Apply a function to rolling windows of the array.
        
        Parameters
        ----------
        window : int
            Size of the rolling window
        func : callable
            Function to apply to each window
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        Parray
            Result of applying the function to rolling windows
        """
        result = np.full(len(self), np.nan)
        
        if self._parallel and len(self) >= self._chunk_size:
            # Process in parallel
            windows = rolling_window(self, window, 1)
            
            # Create arguments for each window
            window_args = [(w, func, args, kwargs) for w in windows]
            
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                results = list(executor.map(_process_chunk, window_args))
            
            result[window-1:] = results
        else:
            # Process sequentially
            for i in range(window-1, len(self)):
                window_data = self[i-window+1:i+1]
                result[i] = func(window_data, *args, **kwargs)
        
        return Parray(result)
    
    
    
