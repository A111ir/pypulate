"""
Parray Module

This module provides a NumPy array extension that supports method chaining
for financial time series analysis, including moving averages and transforms.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any

# Import moving averages
from ..moving_averages import (
    sma, ema, wma, hma, kama, 
    t3, frama, mcginley_dynamic, tma, smma, zlma
)

# Import technical analysis functions
from ..technical import (
    # Momentum indicators
    momentum, roc, rsi, macd, stochastic_oscillator, 
    tsi, williams_r, cci, percent_change, adx,
    
    # Volatility measurements
    bollinger_bands, atr, historical_volatility, 
    keltner_channels, donchian_channels, volatility_ratio,
    
    # Utility functions
    slope, rolling_max, rolling_min, 
    rolling_std, rolling_var, zscore, log, typical_price
)

# Import transforms
from ..transforms.wave import wave, zigzag

# Import filters
from ..filters import (
    kalman_filter, butterworth_filter, savitzky_golay_filter,
    hampel_filter, hodrick_prescott_filter, adaptive_kalman_filter
)



class Parray(np.ndarray):
    """
    A wrapper around numpy arrays that provides method chaining for financial analysis.
    
    This class allows for fluent method chaining like:
    data.ema(9).sma(20)
    
    Examples
    --------
    >>> import numpy as np
    >>> from pypulate.dtypes import Parray
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> ts = Parray(data)
    >>> result = ts.ema(3).sma(2)
    """
    
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
    
    # -------------------------------------------------------------------------
    # Moving Averages
    # -------------------------------------------------------------------------
    
    def sma(self, period=9):
        """Apply Simple Moving Average"""
        return Parray(sma(self, period))
    
    def ema(self, period=9, alpha=None):
        """Apply Exponential Moving Average"""
        return Parray(ema(self, period, alpha))
    
    def wma(self, period=9):
        """Apply Weighted Moving Average"""
        return Parray(wma(self, period))
    
    def tma(self, period=9):
        """Apply Triangular Moving Average"""
        return Parray(tma(self, period))
    
    def smma(self, period=9):
        """Apply Smoothed Moving Average"""
        return Parray(smma(self, period))
    
    def zlma(self, period=9):
        """Apply Zero-Lag Moving Average"""
        return Parray(zlma(self, period))
    
    def hma(self, period=9):
        """Apply Hull Moving Average"""
        return Parray(hma(self, period))
    
    def kama(self, period=9, fast_period=2, slow_period=30):
        """Apply Kaufman Adaptive Moving Average"""
        return Parray(kama(self, period, fast_period, slow_period))
    
    def t3(self, period=9, vfactor=0.7):
        """Apply Tillson T3 Moving Average"""
        return Parray(t3(self, period, vfactor))
    
    def frama(self, period=9, fc_period=198):
        """Apply Fractal Adaptive Moving Average"""
        return Parray(frama(self, period, fc_period))
    
    def mcginley_dynamic(self, period=9, k=0.6):
        """Apply McGinley Dynamic Indicator"""
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
        return Parray(percent_change(self, periods))
    
    def diff(self, periods: int = 1) -> 'Parray':
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
        return Parray(np.diff(self, periods, prepend=np.array([np.nan] * periods)))
    
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
        return Parray(williams_r(self, high, low, period))
    
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
        return Parray(cci(self, period, constant))
    
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
        return Parray(adx(self, period))
    
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
        return Parray(historical_volatility(self, period, annualization_factor))
    
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
        return Parray(atr(self, high, low, period))
    
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
        return Parray(volatility_ratio(self, period, smooth_period))
    
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
        return Parray(typical_price(self, high, low))


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
        return Parray(slope(self, period))
    
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
        return Parray(rolling_max(self, period))
    
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
        return Parray(rolling_min(self, period))
    
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
        return Parray(rolling_std(self, period))
    
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
        return Parray(rolling_var(self, period))

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
        return Parray(zscore(self, period))
    
    def log(self) -> 'Parray':
        """
        Calculate the natural logarithm of the time series.
        
        Returns
        -------
        Parray
            Natural logarithm of the time series
        """
        return Parray(log(self))

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
    
    def zigzag(self, threshold: float = 0.03) -> 'Parray':
        """
        Extract zigzag pivot points from price data based on a percentage threshold.
        
        Parameters
        ----------
        threshold : float, default 0.03
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
        from pypulate.transforms import zigzag as zigzag_func
        result = zigzag_func(self, threshold)
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
        return Parray(kalman_filter(self, process_variance, measurement_variance, 
                                   initial_state, initial_covariance))
    
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
        return Parray(butterworth_filter(self, cutoff, order, filter_type, fs))
    
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
        return Parray(savitzky_golay_filter(self, window_length, polyorder, deriv, delta))
    
    def median_filter(self, kernel_size: int = 3) -> 'Parray':
        """
        Apply a median filter to the time series.
        
        Parameters
        ----------
        kernel_size : int, default 3
            Size of the filter kernel
            
        Returns
        -------
        Parray
            Filtered time series
        """
        from scipy import signal
        return Parray(signal.medfilt(self, kernel_size))
    
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
        return Parray(hampel_filter(self, window_size, n_sigmas))
    
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
        return Parray(adaptive_kalman_filter(self, process_variance_init, measurement_variance_init,
                                           adaptation_rate, window_size, initial_state, initial_covariance))
    
