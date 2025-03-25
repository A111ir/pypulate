"""
Preprocessing Module

This module provides data preprocessing utilities for financial data analysis
without dependencies on pandas, using only numpy and scipy.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats, interpolate
from typing import Union, Tuple, Optional, List



def normalize(data: ArrayLike, method: str = 'l2') -> NDArray[np.float64]:
    """
    Normalize data using vector normalization methods.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    method : str, default='l2'
        Normalization method: 'l1' or 'l2'
        - 'l1': L1 normalization (Manhattan norm)
        - 'l2': L2 normalization (Euclidean norm)
        
    Returns
    -------
    np.ndarray
        Normalized data. For special cases:
        - Empty array: returns empty array
        - All zeros: returns array of zeros
        - Single value: returns array with 1.0
        - All NaN: returns array of NaN
        
    Notes
    -----
    L1 normalization formula: X' = X / sum(|X|)
    L2 normalization formula: X' = X / sqrt(sum(X^2))
    
    Raises
    ------
    ValueError
        If method is not 'l1' or 'l2'
    """
    if method not in ['l1', 'l2']:
        raise ValueError("Method must be one of: 'l1', 'l2'. For min-max scaling, use min_max_scale function.")
    
    data_array = np.array(data, dtype=float)
    
    if len(data_array) == 0:
        return data_array
    
    if not np.any(np.isfinite(data_array)):
        return np.full_like(data_array, np.nan)
    
    if len(data_array) == 1:
        result = np.full_like(data_array, 1.0)
        result[np.isnan(data_array)] = np.nan
        return result
    
    if method == 'l1':
        norm: float = np.nansum(np.abs(data_array))
        if norm < np.finfo(float).tiny: 
            return np.zeros_like(data_array)
        return data_array / norm
    
    else:
        finite_mask = np.isfinite(data_array)
        if not np.any(finite_mask):
            return np.full_like(data_array, np.nan)
        
        squared_sum: float = np.sum(data_array[finite_mask] ** 2)
        if squared_sum < np.finfo(float).tiny: 
            return np.zeros_like(data_array)
        
        result = data_array / np.sqrt(squared_sum)
        result[~finite_mask] = np.nan
        return result


def standardize(data: ArrayLike) -> NDArray[np.float64]:
    """
    Standardize data to have mean 0 and standard deviation 1 (Z-score normalization).
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
        
    Returns
    -------
    np.ndarray
        Standardized data with zero mean and unit variance. For special cases:
        - Empty array: returns empty array
        - Single value: returns array of zeros
        - Constant values: returns array of zeros
        - All NaN: returns array of NaN
    """
    data_array = np.array(data, dtype=float)
    
    if len(data_array) == 0:
        return data_array
        
    if not np.any(np.isfinite(data_array)):
        return data_array
    
    mean = np.nanmean(data_array)
    std = np.nanstd(data_array)
    
    if std < 1e-10:
        return np.zeros_like(data_array)
    
    data_array -= mean
    data_array /= std
    
    return data_array


def min_max_scale(data: ArrayLike, feature_range: Tuple[float, float] = (0, 1)) -> NDArray[np.float64]:
    """
    Scale features to a given range using min-max scaling.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    feature_range : tuple, default=(0, 1)
        Desired range of transformed data
        
    Returns
    -------
    np.ndarray
        Scaled data. For special cases:
        - Empty array: returns empty array
        - Single value: returns array filled with feature_range[0]
        - Constant values: returns array filled with feature_range[0]
        - All NaN: returns array of NaN
        
    Raises
    ------
    ValueError
        If feature_range[0] >= feature_range[1]
    """
    if feature_range[0] >= feature_range[1]:
        raise ValueError("feature_range[0] must be less than feature_range[1]")
    
    data_array = np.array(data, dtype=float)
    
    if len(data_array) == 0:
        return data_array
    
    if not np.any(np.isfinite(data_array)):
        return data_array
    
    min_val : float = np.nanmin(data_array)
    max_val : float = np.nanmax(data_array)
    
    if min_val == max_val:
        return np.full_like(data_array, feature_range[0])
    
    feature_diff = feature_range[1] - feature_range[0]
    
    data_array -= min_val
    data_array /= (max_val - min_val)
    data_array *= feature_diff
    data_array += feature_range[0]
    
    return data_array


def robust_scale(data: ArrayLike, method: str = 'iqr',
                quantile_range: Tuple[float, float] = (25.0, 75.0)) -> NDArray[np.float64]:
    """
    Scale features using statistics that are robust to outliers.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    method : str, default='iqr'
        Method to use for scaling:
        - 'iqr': Use Interquartile Range
        - 'mad': Use Median Absolute Deviation
    quantile_range : tuple, default=(25.0, 75.0)
        Quantile range used to calculate scale when method='iqr'
        
    Returns
    -------
    np.ndarray
        Robustly scaled data. For special cases:
        - Empty array: returns empty array
        - Single value: returns array of zeros
        - Constant values: returns array of zeros
        - All NaN: returns array of NaN
        
    Notes
    -----
    For IQR method: (X - median) / IQR
    For MAD method: (X - median) / (MAD * 1.4826)
    The factor 1.4826 makes the MAD consistent with the standard deviation
    for normally distributed data.
    
    Raises
    ------
    ValueError
        If method is not recognized or if quantile range is invalid
    """
    if method not in ['iqr', 'mad']:
        raise ValueError("Method must be one of: 'iqr', 'mad'")
    
    if quantile_range[0] >= quantile_range[1]:
        raise ValueError("quantile_range[0] must be less than quantile_range[1]")
    
    data_array = np.asarray(data, dtype=float)
    if len(data_array) == 0:
        return data_array
    
    if len(data_array) == 1:
        return np.zeros_like(data_array)
    
    if np.all(np.isnan(data_array)):
        return data_array
    
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    
    if len(valid_data) == 0:
        return data_array
    
    if np.all(valid_data == valid_data[0]):
        result = np.zeros_like(data_array)
        result[~valid_mask] = np.nan
        return result
    
    median = np.median(valid_data)
    
    if method == 'iqr':
        result = np.percentile(valid_data, [float(x) for x in quantile_range], method='linear')
        q1, q3 = result[0], result[1]
        scale = q3 - q1
    else: 
        abs_dev = np.abs(valid_data - median)
        mad = np.median(abs_dev)
        scale = mad * 1.4826
    
    if scale == 0:
        result = np.zeros_like(data_array)
        result[~valid_mask] = np.nan
        return result
    
    result = np.full_like(data_array, np.nan)
    result[valid_mask] = (valid_data - median) / scale
    return result


def quantile_transform(data: ArrayLike, n_quantiles: int = 1000, 
                       output_distribution: str = 'uniform') -> NDArray[np.float64]:
    """
    Transform features using quantile information.
    
    Parameters
    ----------
    data : array-like
        Input data array
    n_quantiles : int, default=1000
        Number of quantiles to use
    output_distribution : str, default='uniform'
        'uniform' or 'normal'
        
    Returns
    -------
    np.ndarray
        Quantile transformed data
    """
    if output_distribution not in ['uniform', 'normal']:
        raise ValueError("output_distribution must be 'uniform' or 'normal'")
    
    data_array = np.asarray(data, dtype=float)
    
    if data_array.size == 0:
        return np.array([], dtype=float)
    
    not_nan = ~np.isnan(data_array)
    indices = np.arange(len(data_array))[not_nan]
    valid_data = data_array[not_nan]
    
    if len(valid_data) == 0:
        return np.full_like(data_array, np.nan)
    
    if len(valid_data) == 1:
        result = np.full_like(data_array, np.nan)
        result[indices] = 0.5  
        if output_distribution == 'normal':
            result[indices] = 0.0  
        return result
    
    ranks = stats.rankdata(valid_data, method='average')
    
    if len(valid_data) > n_quantiles:
        n_q = max(2, n_quantiles)
        ranks_scaled = (ranks - 1) * (n_q - 1) / (len(ranks) - 1)
        ranks_scaled = np.round(ranks_scaled).astype(int)
        ranks_scaled = ranks_scaled / (n_q - 1)
    else:
        ranks_scaled = (ranks - 0.5) / len(ranks)
    
    if output_distribution == 'normal':
        ranks_scaled = np.clip(ranks_scaled, 0.001, 0.999)
        output = stats.norm.ppf(ranks_scaled)
    else:  
        output = ranks_scaled
    
    result = np.full_like(data_array, np.nan)
    result[indices] = output
    
    return result


def winsorize(data: ArrayLike, limits: Union[float, Tuple[float, float]] = 0.05) -> NDArray[np.float64]:
    """
    Limit extreme values in data.
    
    Parameters
    ----------
    data : array-like
        Input data array
    limits : float or tuple, default=0.05
        If a float, it is the proportion to cut on each side.
        If a tuple of two floats, they represent the proportions to cut from the lower and upper bounds.
        
    Returns
    -------
    np.ndarray
        Winsorized data
    """
    if isinstance(limits, float):
        limits = (limits, limits)
    
    data_array = np.asarray(data, dtype=float)
    
    if data_array.size == 0:
        return np.array([], dtype=float)
    
    not_nan = ~np.isnan(data_array)
    valid_data = data_array[not_nan]
    
    if len(valid_data) == 0:
        return np.full_like(data_array, np.nan)
    
    if len(valid_data) == 1:
        return data_array.copy()
    
    lower_limit = np.nanpercentile(valid_data, limits[0] * 100)
    upper_limit = np.nanpercentile(valid_data, 100 - limits[1] * 100)
    
    result = np.copy(data_array)
    result[result < lower_limit] = lower_limit
    result[result > upper_limit] = upper_limit
    
    return result


def remove_outliers(data: ArrayLike, method: str = 'zscore', threshold: float = 2.0) -> NDArray[np.float64]:
    """
    Remove outliers from data by replacing them with NaN.
    
    Parameters
    ----------
    data : array-like
        Input data array
    method : str, default='zscore'
        Method to detect outliers: 'zscore', 'iqr', or 'mad'
    threshold : float, default=2.0
        Threshold for outlier detection:
        - For 'zscore': number of standard deviations
        - For 'iqr': multiplier of IQR
        - For 'mad': multiplier of MAD
        
    Returns
    -------
    np.ndarray
        Data with outliers replaced by NaN. Original NaN values remain NaN.
    """
    data_array = np.asarray(data, dtype=float)
    
    if data_array.size == 0:
        return np.array([], dtype=float)
    
    result = np.copy(data_array)
    valid_mask = ~np.isnan(result)
    valid_data = data_array[valid_mask]
    
    if len(valid_data) == 0:
        return result
    
    if method == 'zscore':
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median)) * 1.4826
        if mad == 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            if std == 0:
                return result
            z_scores = np.abs((data_array - mean) / std)
        else:
            z_scores = np.abs((data_array - median) / mad)
        result[valid_mask & (z_scores > threshold)] = np.nan
    
    elif method == 'iqr':
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        if iqr == 0:
            return result
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        result[valid_mask & ((data_array < lower_bound) | (data_array > upper_bound))] = np.nan
    
    elif method == 'mad':
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))
        mad = mad * 1.4826  
        if mad == 0:
            return result
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
        result[valid_mask & ((data_array < lower_bound) | (data_array > upper_bound))] = np.nan
    
    else:
        raise ValueError("Method must be one of: 'zscore', 'iqr', 'mad'")
    
    return result


def fill_missing(data: ArrayLike, method: str = 'mean', value: Optional[float] = None) -> NDArray[np.float64]:
    """
    Fill missing values in data using efficient NumPy operations.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    method : str, default='mean'
        Method to fill missing values: 'mean', 'median', 'mode', 'forward', 'backward', 'value'
    value : float, optional
        Value to use when method='value'
        
    Returns
    -------
    np.ndarray
        Data with missing values filled. For all-NaN input:
        - Statistical methods (mean, median, mode) return all-NaN array
        - Forward/backward fill return all-NaN array
        - Value fill returns array filled with specified value
        
    Raises
    ------
    ValueError
        - If method is not recognized
        - If method='value' but no value is provided
    """
    valid_methods = ['mean', 'median', 'mode', 'forward', 'backward', 'value']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of: {', '.join(valid_methods)}")
    
    if method == 'value' and value is None:
        raise ValueError("Value must be provided when method='value'")
    
    result = np.array(data, dtype=float)
    
    if not np.any(np.isnan(result)):
        return result
    
    if np.all(np.isnan(result)):
        if method == 'value':
            result.fill(value)
        return result
    
    nan_mask = np.isnan(result)
    valid_mask = ~nan_mask
    
    if method == 'mean':
        fill_value = np.mean(result[valid_mask])
        result[nan_mask] = fill_value
    
    elif method == 'median':
        fill_value = np.median(result[valid_mask])
        result[nan_mask] = fill_value
    
    elif method == 'mode':
        valid_data = result[valid_mask]
        unique_vals, counts = np.unique(valid_data, return_counts=True)
        mode_idx = np.argmax(counts)
        result[nan_mask] = unique_vals[mode_idx]
    
    elif method == 'forward':
        valid_indices = np.where(valid_mask)[0]
        nan_indices = np.where(nan_mask)[0]
        
        fill_idx = np.asarray(np.searchsorted(valid_indices, nan_indices) - 1)
        mask = fill_idx >= 0
        fill_idx_masked = fill_idx[mask]
        result[nan_indices[mask]] = result[valid_indices[fill_idx_masked]]
    
    elif method == 'backward':
        valid_indices = np.where(valid_mask)[0]
        nan_indices = np.where(nan_mask)[0]
        
        fill_idx = np.asarray(np.searchsorted(valid_indices, nan_indices))
        mask = fill_idx < len(valid_indices)
        fill_idx_masked = fill_idx[mask]
        result[nan_indices[mask]] = result[valid_indices[fill_idx_masked]]
    
    else:
        result[nan_mask] = value
    
    return result


def interpolate_missing(data: ArrayLike, method: str = 'linear') -> NDArray[np.float64]:
    """
    Interpolate missing values in data using efficient NumPy operations.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    method : str, default='linear'
        Interpolation method: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        
    Returns
    -------
    np.ndarray
        Data with missing values interpolated. For missing values at the start or end:
        - 'nearest' and 'zero' methods will use the nearest valid value
        - other methods will leave them as NaN
        
    """
    valid_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of: {', '.join(valid_methods)}")
    
    data_array = np.array(data, dtype=float)
    
    if not np.any(np.isnan(data_array)):
        return data_array
    
    result = np.copy(data_array)
    nan_mask = np.isnan(result)
    
    valid_indices = np.nonzero(~nan_mask)[0]
    
    if len(valid_indices) < 2:
        return result
    
    valid_values = result[valid_indices]
    
    if method in ['nearest', 'zero']:
        f = interpolate.interp1d(
            valid_indices, 
            valid_values,
            kind=method,
            bounds_error=False,
            fill_value=(valid_values[0], valid_values[-1])
        )
        result[nan_mask] = f(np.nonzero(nan_mask)[0])
    else:
        f = interpolate.interp1d(
            valid_indices,
            valid_values,
            kind=method,
            bounds_error=False,
            fill_value=np.nan
        )
        missing_indices = np.nonzero(nan_mask)[0]
        in_range = (missing_indices >= valid_indices[0]) & (missing_indices <= valid_indices[-1])
        result[missing_indices[in_range]] = f(missing_indices[in_range])
    
    return result


def resample(data: ArrayLike, factor: int, method: str = 'mean') -> NDArray[np.float64]:
    """
    Resample data by aggregating values using efficient NumPy operations.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    factor : int
        Resampling factor (e.g., 5 means aggregate every 5 points)
    method : str, default='mean'
        Aggregation method: 'mean', 'median', 'sum', 'min', 'max'
        Note: For groups containing all NaN values:
        - sum will return NaN
        - mean, median, min, max will return NaN
        
    Returns
    -------
    np.ndarray
        Resampled data. Length will be floor(len(data)/factor).
        Remaining data points that don't fill a complete group are discarded.
    """
    if factor <= 1:
        return np.array(data, dtype=float)
    
    data_array = np.array(data, dtype=float)
    n_complete = len(data_array) // factor
    
    if n_complete == 0:
        return np.array([])
    
    reshaped = data_array[:n_complete * factor].reshape(n_complete, factor)
    
    if method == 'mean':
        all_nan_mask = np.all(np.isnan(reshaped), axis=1)
        
        results = np.zeros(n_complete)
        for i in range(n_complete):
            if all_nan_mask[i]:
                results[i] = np.nan
            else:
                valid_data = reshaped[i][~np.isnan(reshaped[i])]
                if len(valid_data) > 0:
                    results[i] = np.mean(valid_data)
                else:
                    results[i] = np.nan
        
        return results
    elif method == 'median':
        return np.nanmedian(reshaped, axis=1)
    elif method == 'sum':
        sums = np.nansum(reshaped, axis=1)
        all_nan_mask = np.all(np.isnan(reshaped), axis=1)
        sums[all_nan_mask] = np.nan
        return sums
    elif method == 'min':
        return np.nanmin(reshaped, axis=1)
    elif method == 'max':
        return np.nanmax(reshaped, axis=1)
    else:
        raise ValueError("Method must be one of: 'mean', 'median', 'sum', 'min', 'max'")


def rolling_window(data: ArrayLike, window_size: int, step: int = 1) -> NDArray[np.float64]:
    """
    Create rolling windows of data using efficient NumPy striding.
    
    Parameters
    ----------
    data : array-like
        Input data array
    window_size : int
        Size of the rolling window
    step : int, default=1
        Step size between windows
        
    Returns
    -------
    np.ndarray
        Array of rolling windows. Shape will be (n_windows, window_size)
        where n_windows = max(0, (len(data) - window_size) // step + 1)
        
    Raises
    ------
    ValueError
        If window_size or step is not positive
    """
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    
    data_array = np.asarray(data, dtype=float)
    n_samples = len(data_array)
    
    if n_samples < window_size:
        return np.zeros((0, window_size))
    
    n_windows = (n_samples - window_size) // step + 1
    
    if n_windows <= 0:
        if n_samples == window_size:
            return data_array.reshape(1, -1)
        return np.zeros((0, window_size))
    
    new_shape = (n_windows, window_size)
    
    new_strides = (step * data_array.strides[0], data_array.strides[0])
    
    windows = np.lib.stride_tricks.as_strided(
        data_array,
        shape=new_shape,
        strides=new_strides,
        writeable=False
    )
    
    return windows.copy()  # Return a copy to ensure memory safety


def lag_features(data: ArrayLike, lags: List[int]) -> NDArray[np.float64]:
    """
    Create lagged features from data using vectorized operations.
    
    Parameters
    ----------
    data : array-like
        Input data array
    lags : list of int
        List of lag values. Zero lag returns the original values,
        negative lags are ignored, and lags larger than the data length
        result in NaN columns.
        
    Returns
    -------
    np.ndarray
        Array with original and lagged features as columns.
        First column is the original data, followed by columns for each lag.
        NaN values are used for undefined lag positions.
    """
    data_array = np.asarray(data, dtype=float)
    n_samples = len(data_array)
    n_features = 1 + len(lags)
    
    result = np.full((n_samples, n_features), np.nan)
    
    result[:, 0] = data_array
    
    valid_lags = np.array([lag for lag in lags if lag >= 0])
    valid_indices = np.array([i + 1 for i, lag in enumerate(lags) if lag >= 0])
    
    if len(valid_lags) == 0:
        return result

    zero_mask = valid_lags == 0
    if np.any(zero_mask):
        result[:, valid_indices[zero_mask]] = data_array[:, np.newaxis]
    
    pos_mask = valid_lags > 0
    if np.any(pos_mask):
        pos_lags = valid_lags[pos_mask]
        pos_indices = valid_indices[pos_mask]
        
        for lag, idx in zip(pos_lags, pos_indices):
            result[lag:, idx] = data_array[:-lag]
    
    return result


def difference(data: ArrayLike, order: int = 1) -> NDArray[np.float64]:
    """
    Calculate differences between consecutive elements of a time series.
    
    Parameters
    ----------
    data : array-like
        Input data array
    order : int, default=1
        The order of differencing. Must be non-negative.
        
    Returns
    -------
    np.ndarray
        Array of differences with length n-order, where n is the length of the input array.
        For order=0, returns the original array.
        NaN values in input result in NaN differences only where NaN values are involved.
        
    Raises
    ------
    ValueError
        If order is negative or larger than the length of the data.
    """
    if order < 0:
        raise ValueError("Order must be non-negative")
    
    data_array = np.asarray(data, dtype=float)
    
    if len(data_array) == 0:
        return data_array
    
    if order > len(data_array):
        raise ValueError("Order cannot be larger than the length of the data")
    
    if order == 0:
        return data_array
    
    return np.diff(data_array, n=order)


def log_transform(data: ArrayLike, base: Optional[float] = None, 
                 offset: float = 0.0) -> NDArray[np.float64]:
    """
    Apply logarithmic transformation to data.
    
    Parameters
    ----------
    data : array-like
        Input data array, can be list or numpy array
    base : float, optional
        Base of logarithm. If None, natural logarithm is used.
        Common bases: None (natural log), 2, 10
    offset : float, default=0.0
        Offset added to data before taking logarithm (useful for non-positive data)
        
    Returns
    -------
    np.ndarray
        Log-transformed data. For special cases:
        - Empty array: returns empty array
        - Single value: returns log of (value + offset)
        - All NaN: returns array of NaN
        - Non-positive values: raises ValueError if any (value + offset) <= 0
    
    Raises
    ------
    ValueError
        If any value after offset is non-positive
    """
    # Convert to array efficiently and ensure float64 for numerical stability
    data_array = np.asarray(data, dtype=np.float64)
    
    # Early return for empty arrays
    if len(data_array) == 0:
        return data_array
    
    # Handle all-NaN arrays efficiently
    if np.all(np.isnan(data_array)):
        return data_array.copy()
    
    # Add offset efficiently using in-place operation
    if offset != 0.0:
        data_array = data_array + offset
    
    # Check for non-positive values efficiently using valid data only
    valid_mask = ~np.isnan(data_array)
    if np.any(data_array[valid_mask] <= 0):
        raise ValueError("Data contains non-positive values. Use a larger offset.")
    
    # Pre-allocate result array for NaN preservation
    result = np.full_like(data_array, np.nan)
    
    # Apply log transform efficiently only to valid data
    if base is None:
        result[valid_mask] = np.log(data_array[valid_mask])
    elif base == 10:
        result[valid_mask] = np.log10(data_array[valid_mask])
    elif base == 2:
        result[valid_mask] = np.log2(data_array[valid_mask])
    else:
        # Use more stable method for custom base
        log_base = np.log(base)
        result[valid_mask] = np.log(data_array[valid_mask]) / log_base
    
    return result


def power_transform(data: ArrayLike, method: str = 'yeo-johnson', 
                    standardize: bool = True) -> NDArray[np.float64]:
    """
    Apply power transformation (Box-Cox or Yeo-Johnson) to make data more Gaussian-like.
    
    Parameters
    ----------
    data : array-like
        Input data array, can be list or numpy array
    method : str, default='yeo-johnson'
        The power transform method:
        - 'box-cox': only works with positive values
        - 'yeo-johnson': works with both positive and negative values
    standardize : bool, default=True
        Whether to standardize the data after transformation
        
    Returns
    -------
    np.ndarray
        Power transformed data. NaN values in input remain NaN in output.
        For special cases:
        - Empty array: returns empty array
        - Single value: returns array of zeros if standardize=True,
          or the log1p of the constant if standardize=False
        - Constant values: returns array of zeros if standardize=True,
          or the log1p of the constant if standardize=False
        - All NaN: returns array of NaN
        
    Raises
    ------
    ValueError
        - If method is not one of 'box-cox', 'yeo-johnson'
    """
    if method not in {'box-cox', 'yeo-johnson'}:
        raise ValueError("Method must be one of: 'box-cox', 'yeo-johnson'")
    
    data_array = np.asarray(data, dtype=np.float64)
    
    if len(data_array) == 0:
        return data_array
    
    nan_mask = np.isnan(data_array)
    valid_mask = ~nan_mask
    
    if not np.any(valid_mask):
        return data_array.copy()
    
    valid_data = data_array[valid_mask]
    
    unique_values = np.unique(valid_data)
    if len(unique_values) == 1:
        result = np.full_like(data_array, np.nan)
        if standardize:
            result[valid_mask] = 0.0
        else:
            constant = unique_values[0]
            result[valid_mask] = np.log1p(np.abs(constant)) * np.sign(constant)
        return result
    
    result = np.full_like(data_array, np.nan)
    
    if method == 'box-cox':
        if np.any(valid_data <= 0):
            raise ValueError("Box-Cox transformation requires strictly positive values")
        
        transformed, _ = stats.boxcox(valid_data)
        result[valid_mask] = transformed
    
    else:  
        transformed = np.zeros_like(valid_data)
        
        pos_mask = valid_data >= 0
        neg_mask = ~pos_mask
        
        if np.any(pos_mask):
            transformed[pos_mask] = np.log1p(valid_data[pos_mask])
        if np.any(neg_mask):
            transformed[neg_mask] = -np.log1p(-valid_data[neg_mask])
        
        result[valid_mask] = transformed
    
    if standardize and np.any(valid_mask):
        valid_transformed = result[valid_mask]
        std = np.std(valid_transformed)
        
        if std > np.finfo(np.float64).tiny:
            mean = np.mean(valid_transformed)
            result[valid_mask] = (valid_transformed - mean) / std
        else:
            result[valid_mask] = 0.0
    
    return result


def scale_to_range(data: ArrayLike, 
                  feature_range: Tuple[float, float] = (0.0, 1.0)) -> NDArray[np.float64]:
    """
    Scale data to a specific range while preserving relative distances.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    feature_range : tuple, default=(0.0, 1.0)
        Desired range for transformed data (min, max)
        
    Returns
    -------
    np.ndarray
        Data scaled to target range. For special cases:
        - Empty array: returns empty array
        - Single value: returns array filled with feature_range[0]
        - Constant values: returns array filled with feature_range[0]
        - All NaN: returns array of NaN
        
    Raises
    ------
    ValueError
        If feature_range[0] >= feature_range[1]
    """
    if feature_range[0] >= feature_range[1]:
        raise ValueError("feature_range[0] must be less than feature_range[1]")
    
    data_array = np.asarray(data, dtype=np.float64)
    
    if len(data_array) == 0:
        return data_array
    
    if np.all(np.isnan(data_array)):
        return data_array.copy()
    
    if len(data_array) == 1:
        return np.full_like(data_array, feature_range[0])
    
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    
    if len(np.unique(valid_data)) == 1:
        result = np.full_like(data_array, feature_range[0])
        result[~valid_mask] = np.nan
        return result
    
    current_min: np.float64 = np.min(valid_data)
    current_max: np.float64 = np.max(valid_data)
    
    data_range = current_max - current_min
    feature_diff = feature_range[1] - feature_range[0]
    
    result = np.full_like(data_array, np.nan)
    result[valid_mask] = (valid_data - current_min) / data_range * feature_diff + feature_range[0]
    
    return result


def clip_outliers(data: ArrayLike, lower_percentile: float = 1.0, 
                 upper_percentile: float = 99.0) -> NDArray[np.float64]:
    """
    Clip values outside specified percentiles.
    
    Parameters
    ----------
    data : array-like
        Input data array
    lower_percentile : float, default=1.0
        Lower percentile (between 0 and 100)
    upper_percentile : float, default=99.0
        Upper percentile (between 0 and 100)
        
    Returns
    -------
    np.ndarray
        Data with outliers clipped. For special cases:
        - Empty array: returns empty array
        - Single value: returns unchanged value
        - Constant values: returns unchanged array
        - All NaN: returns array of NaN
        
    Raises
    ------
    ValueError
        If percentiles are not between 0 and 100 or if lower_percentile > upper_percentile
    """
    if not 0 <= lower_percentile <= 100:
        raise ValueError("lower_percentile must be between 0 and 100")
    if not 0 <= upper_percentile <= 100:
        raise ValueError("upper_percentile must be less than or equal to 100")
    if lower_percentile > upper_percentile:
        raise ValueError("lower_percentile must be less than or equal to upper_percentile")
    
    data_array = np.asarray(data, dtype=np.float64)
    
    if len(data_array) == 0:
        return data_array
    
    if np.all(np.isnan(data_array)):
        return data_array.copy()
    
    if len(data_array) == 1:
        return data_array.copy()
    
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    
    if len(np.unique(valid_data)) == 1:
        return data_array.copy()
    
    lower_bound = np.percentile(valid_data, lower_percentile)
    upper_bound = np.percentile(valid_data, upper_percentile)
    
    result = data_array.copy()
    result[valid_mask] = np.clip(valid_data, lower_bound, upper_bound)
    
    return result


def _kmeans_binning(data: ArrayLike, n_bins: int) -> NDArray[np.float64]:
    """
    Vectorized k-means binning implementation using only numpy.
    
    Parameters
    ----------
    data : array-like
        Input data array
    n_bins : int
        Number of bins
        
    Returns
    -------
    np.ndarray
        Array of bin labels (1 to n_bins)
    """
    data_array = np.array(data, dtype=float)
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    
    if len(valid_data) == 0:
        return np.full_like(data_array, np.nan)
    
    # Initialize centroids using linspace for even distribution
    min_val: float = np.min(valid_data)
    max_val: float = np.max(valid_data)
    centroids = np.linspace(min_val, max_val, n_bins)
    
    # Handle edge case of all identical values
    if np.isclose(min_val, max_val):
        result = np.full_like(data_array, np.nan)
        result[valid_mask] = 1.0  # Assign all to first bin
        return result
    
    max_iter = 100
    tol = 1e-4
    
    # Pre-allocate arrays for efficiency
    labels = np.zeros(len(valid_data), dtype=int)
    
    for _ in range(max_iter):
        # Compute distances matrix in one vectorized operation
        # Shape: (n_samples, n_bins)
        distances = np.abs(valid_data[:, np.newaxis] - centroids)
        
        # Assign points to nearest centroid
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
            
        labels = new_labels
        
        # Update centroids using vectorized operations
        new_centroids = np.zeros_like(centroids)
        for i in range(n_bins):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = np.mean(valid_data[mask])
            else:
                new_centroids[i] = centroids[i]  # Keep old centroid if no points assigned
        
        # Check if centroids have converged
        if np.allclose(centroids, new_centroids, rtol=tol, atol=tol):
            break
            
        centroids = new_centroids
    
    # Assign labels to original data points (including NaNs)
    result = np.full_like(data_array, np.nan)
    
    # Compute distances for all valid data points in one operation
    distances = np.abs(data_array[valid_mask][:, np.newaxis] - centroids)
    
    # Assign 1-based bin labels
    result[valid_mask] = np.argmin(distances, axis=1) + 1
    
    return result


def discretize(data: ArrayLike, 
               n_bins: int = 5, 
               strategy: str = 'uniform') -> NDArray[np.float64]:
    """
    Discretize continuous data into bins using efficient vectorized operations.
    
    Parameters
    ----------
    data : array-like
        Input data array to be discretized. Will be flattened if multi-dimensional.
    n_bins : int, default=5
        Number of bins to create. Must be positive.
    strategy : str, default='uniform'
        Strategy to use for creating bins:
        - 'uniform': Equal-width bins
        - 'quantile': Equal-frequency bins
        - 'kmeans': Bins based on k-means clustering
        
    Returns
    -------
    np.ndarray
        Array of bin labels (1 to n_bins). NaN values in input remain NaN in output.
        For special cases:
        - Empty array: returns empty array
        - Single value or constant array: returns array filled with 1.0 (NaN preserved)
        - All NaN: returns array of NaN
        
    Notes
    -----
    - Uses efficient vectorized operations for binning
    - Handles NaN values gracefully
    - Memory efficient implementation avoiding unnecessary copies
    - Bin labels are 1-based (1 to n_bins)
    
    Raises
    ------
    ValueError
        - If strategy is not one of 'uniform', 'quantile', or 'kmeans'
        - If n_bins is not positive
    """
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    
    if strategy not in {'uniform', 'quantile', 'kmeans'}:
        raise ValueError("strategy must be one of 'uniform', 'quantile', or 'kmeans'")
    
    # Convert to array efficiently
    data_array: NDArray[np.float64] = np.asarray(data, dtype=np.float64)
    
    if len(data_array) == 0:
        return np.array([], dtype=np.float64)
    
    # Handle NaN values efficiently
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    
    if len(valid_data) == 0:
        return data_array.copy()  # Return array of NaN
    
    # Check for constant input
    if len(np.unique(valid_data)) == 1:
        result = np.full_like(data_array, 1.0)
        result[~valid_mask] = np.nan
        return result
    
    # Pre-allocate result array
    result = np.full_like(data_array, np.nan)
    
    if strategy == 'uniform':
        # Compute bin edges efficiently
        edges = np.linspace(
            np.min(valid_data),
            np.max(valid_data) * (1 + np.finfo(float).eps),  # Ensure max value falls in last bin
            n_bins + 1
        )
        result[valid_mask] = np.digitize(valid_data, edges[:-1])
        
    elif strategy == 'quantile':
        # Compute quantile bin edges efficiently
        edges = np.percentile(
            valid_data,
            np.linspace(0, 100, n_bins + 1),
            method='linear'
        )
        # Ensure unique edges for proper binning
        edges = np.unique(edges)
        if len(edges) < n_bins + 1:
            # Handle case where we have fewer unique edges than bins
            edges = np.linspace(edges[0], edges[-1], n_bins + 1)
        result[valid_mask] = np.digitize(valid_data, edges[:-1])
        
    else:  # kmeans
        # Use optimized k-means implementation
        result[valid_mask] = _kmeans_binning(valid_data, n_bins)[~np.isnan(valid_data)]
    
    return result


def polynomial_features(data: ArrayLike, degree: int = 2) -> NDArray[np.float64]:
    """
    Generate polynomial features up to specified degree.
    
    Parameters
    ----------
    data : array-like
        Input data array. Will be flattened if multi-dimensional.
    degree : int, default=2
        Maximum degree of polynomial features. Must be a positive integer.
        
    Returns
    -------
    np.ndarray
        Array with polynomial features as columns:
        - First column contains 1s (bias term)
        - Subsequent columns contain increasing powers (x, x², ..., x^degree)
        Shape will be (n_samples, degree + 1)
        
    Notes
    -----
    - Uses efficient vectorized operations for polynomial computation
    - Handles NaN values gracefully (propagates through powers)
    - Memory efficient implementation avoiding unnecessary copies
    
    Raises
    ------
    ValueError
        If degree is not a positive integer
    """
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    
    data_array: NDArray[np.float64] = np.asarray(data, dtype=np.float64).ravel()
    n_samples: int = len(data_array)
    
    if n_samples == 0:
        return np.zeros((0, degree + 1), dtype=np.float64)
    
    result: NDArray[np.float64] = np.empty((n_samples, degree + 1), dtype=np.float64)
    result[:, 0] = 1.0  
    
    if degree >= 1:
        result[:, 1] = data_array  
        
        if degree >= 2:
            data_col = data_array.reshape(-1, 1)
            result[:, 2:] = np.power(data_col, np.arange(2, degree + 1))
    
    return result


def dynamic_tanh(data: ArrayLike, alpha: float = 1.0) -> NDArray[np.float64]:
    """
    Apply Dynamic Tanh (DyT) transformation to data, which helps normalize data 
    while preserving relative differences and handling outliers well.
    
    Parameters
    ----------
    data : array-like
        Input data array, can contain None or np.nan as missing values
    alpha : float, default=1.0
        Scaling factor that controls the transformation intensity.
        Higher values lead to more aggressive normalization (less extreme values).
        
    Returns
    -------
    np.ndarray
        DyT-transformed data with values in range (-1, 1). For special cases:
        - Empty array: returns empty array
        - Single value: returns array of zeros
        - Constant values: returns array of zeros
        - All NaN: returns array of NaN
        
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
    data_array = np.asarray(data, dtype=np.float64)
    
    if len(data_array) == 0:
        return data_array
    
    if np.all(np.isnan(data_array)):
        return data_array.copy()
    
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    
    if len(valid_data) == 0:
        return data_array.copy()
    
    # For single value or constant data, return zeros (like other normalization methods)
    if len(np.unique(valid_data)) == 1:
        result = np.zeros_like(data_array)
        result[~valid_mask] = np.nan
        return result
    
    # Step 1: Center data using median (robust to outliers)
    median = np.median(valid_data)
    centered_data = data_array - median
    
    # Step 2: Scale data using MAD (Median Absolute Deviation)
    # MAD is a robust measure of dispersion that's less affected by outliers
    mad = np.median(np.abs(valid_data - median))
    
    # Apply small epsilon to prevent division by zero
    epsilon = np.finfo(np.float64).eps
    # Corrected: higher alpha = less extreme values (more aggressive scaling)
    # Multiply MAD by alpha, so higher alpha means more scaling
    scale_factor = mad * alpha
    
    if scale_factor < epsilon:
        # Handle near-constant data with tiny variation
        scale_factor = np.std(valid_data) * alpha
        if scale_factor < epsilon:
            result = np.zeros_like(data_array)
            result[~valid_mask] = np.nan
            return result
    
    scaled_data = centered_data / scale_factor
    
    # Step 3: Apply tanh transformation to squash values to (-1, 1) range
    # tanh preserves the sign and relative magnitude while bounding values
    result = np.tanh(scaled_data)
    
    return result

