"""
Statistics Module

This module provides statistical analysis utilities for financial data analysis.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import stats, signal
from typing import Union, Tuple, Optional, List, Dict


def descriptive_stats(data: ArrayLike) -> Dict[str, float]:
    """
    Calculate descriptive statistics for data efficiently.
    
    Parameters
    ----------
    data : array-like
        Input data array. Must be 1-dimensional.
    
    Returns
    -------
    dict
        Dictionary containing the following statistics:
        - count: Number of non-NaN values
        - mean: Arithmetic mean
        - std: Standard deviation (N-1)
        - min: Minimum value
        - q1: First quartile (25th percentile)
        - median: Median (50th percentile)
        - q3: Third quartile (75th percentile)
        - max: Maximum value
        - skewness: Sample skewness
        - kurtosis: Sample excess kurtosis
        
    Raises
    ------
    ValueError: If input data is not a 1D array
    """
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 1:
        raise ValueError("Input data must be a 1D array")
    
    stats_dict = {
        'count': 0,
        'mean': np.nan,
        'std': np.nan,
        'min': np.nan,
        'q1': np.nan,
        'median': np.nan,
        'q3': np.nan,
        'max': np.nan,
        'skewness': np.nan,
        'kurtosis': np.nan
    }
    
    valid_mask = ~np.isnan(data_array)
    valid_count: int = int(np.sum(valid_mask))
    stats_dict['count'] = valid_count
    
    if valid_count == 0:
        return stats_dict
    
    valid_data = data_array[valid_mask]
    
    if valid_count == 1:
        value = valid_data[0]
        stats_dict.update({
            'mean': value,
            'median': value,
            'min': value,
            'q1': value,
            'q3': value,
            'max': value
        })
        return stats_dict
    
    if np.allclose(valid_data, valid_data[0], rtol=1e-14, atol=1e-14):
        value = float(valid_data[0])
        stats_dict.update({
            'mean': value,
            'median': value,
            'min': value,
            'q1': value,
            'q3': value,
            'max': value,
            'std': 0.0
        })
        return stats_dict
    
    stats_dict.update({
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data, ddof=1))
    })
    
    
    quartiles: NDArray[np.float64] = np.percentile(valid_data, [25, 50, 75], method='linear')
    stats_dict.update({
        'q1': float(quartiles[0]),
        'median': float(quartiles[1]),
        'q3': float(quartiles[2])
    })
    
    if valid_count >= 3:
        mean_var = stats.describe(valid_data)
        stats_dict['skewness'] = float(mean_var.skewness)
        stats_dict['kurtosis'] = float(mean_var.kurtosis)
    
    return stats_dict


def correlation_matrix(data: ArrayLike, method: str = 'pearson', min_periods: int = 1) -> NDArray[np.float64]:
    """Calculate the correlation matrix for a 2D array.

    Parameters
    ----------
    data : ArrayLike
        2D array of shape (n_samples, n_features)
    method : str, optional
        The correlation method to use ('pearson', 'spearman', or 'kendall')
        Default is 'pearson'
    min_periods : int, optional
        Minimum number of valid observations required for each pair of columns
        Default is 1

    Returns
    -------
    NDArray[np.float64]
        Correlation matrix of shape (n_features, n_features)
        
    Raises
    ------
    ValueError: If input data is not a 2D array or method is not one of 'pearson', 'spearman', 'kendall'
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("Method must be one of: 'pearson', 'spearman', 'kendall'")

    data_array = np.asarray(data, dtype=np.float64)
    n_samples, n_features = data_array.shape
    result = np.eye(n_features, dtype=np.float64)

    if n_samples == 1:
        return result

    if method == 'kendall':
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mask = ~(np.isnan(data_array[:, i]) | np.isnan(data_array[:, j]))
                n_valid: int = np.sum(mask)
                if n_valid >= min_periods:
                    tau, _ = stats.kendalltau(data_array[mask, i], data_array[mask, j])
                    result[i, j] = result[j, i] = np.nan if np.isnan(tau) else tau
        return result

    if method == 'spearman':
        data_array = np.array([
            stats.rankdata(col[~np.isnan(col)]) if len(col[~np.isnan(col)]) > 0 
            else np.full_like(col, np.nan) for col in data_array.T
        ]).T

    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(data_array[:, i]) | np.isnan(data_array[:, j]))
            n_valid_pair: int = np.sum(mask)
            
            if n_valid_pair < min_periods:
                result[i, j] = result[j, i] = np.nan
                continue

            x = data_array[mask, i]
            y = data_array[mask, j]

            x_std = np.std(x, ddof=1)
            y_std = np.std(y, ddof=1)
            
            if np.isclose(x_std, 0) or np.isclose(y_std, 0):
                if np.allclose(x, y):
                    result[i, j] = result[j, i] = 1.0
                else:
                    result[i, j] = result[j, i] = np.nan
                continue

            corr_matrix = np.corrcoef(x, y)
            result[i, j] = result[j, i] = corr_matrix[0, 1]

    return result


def covariance_matrix(data: ArrayLike, ddof: int = 1) -> NDArray[np.float64]:
    """Calculate covariance matrix for multivariate data.
    
    Parameters
    ----------
    data : ArrayLike
        Input data array with shape (n_samples, n_features).
        Can contain None or np.nan as missing values.
    ddof : int, default=1
        Delta degrees of freedom for the covariance calculation.
        The divisor used in calculations is N - ddof, where N represents
        the number of non-missing elements.
        
    Returns
    -------
    NDArray[np.float64]
        Covariance matrix with shape (n_features, n_features).
        For features i and j, the element [i,j] represents their covariance.
        The matrix is symmetric, with variances on the diagonal.
        
    Raises
    ------
    ValueError: If input data is not a 2D array
    """
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    
    n_samples, n_features = data_array.shape
    
    if n_samples <= ddof:
        return np.full((n_features, n_features), np.nan)
    
    result = np.zeros((n_features, n_features), dtype=np.float64)
    
    scales = np.ones(n_features, dtype=np.float64)
    means = np.zeros(n_features, dtype=np.float64)
    valid_counts = np.zeros(n_features, dtype=np.int64)
    
    for i in range(n_features):
        valid_mask = ~np.isnan(data_array[:, i])
        valid_data = data_array[valid_mask, i]
        valid_counts[i] = np.sum(valid_mask)
        
        if valid_counts[i] > ddof:
            means[i] = np.mean(valid_data)
            abs_max : float = np.max(np.abs(valid_data))
            if abs_max >= 1.0:
                scales[i] = abs_max
            else:
                nonzero_min = np.min(np.abs(valid_data[valid_data != 0])) if np.any(valid_data != 0) else 1.0
                scales[i] = nonzero_min
    
    for i in range(n_features):
        for j in range(i + 1):
            mask = ~(np.isnan(data_array[:, i]) | np.isnan(data_array[:, j]))
            n_valid_pair: int = np.sum(mask)
            
            if n_valid_pair <= ddof:
                result[i, j] = result[j, i] = np.nan
                continue
            
            x = (data_array[mask, i] - means[i]) / scales[i]
            y = (data_array[mask, j] - means[j]) / scales[j]
            
            cov = np.sum(x * y) / (n_valid_pair - ddof)
            
            cov *= scales[i] * scales[j]
            
            result[i, j] = result[j, i] = cov
    
    return result


def autocorrelation(data: ArrayLike, max_lag: int = 20) -> NDArray[np.float64]:
    """Calculate autocorrelation function (ACF) for time series data.
    
    The autocorrelation function measures the correlation between observations 
    at different time lags. It helps identify patterns and seasonality in time series data.
    
    Parameters
    ----------
    data : ArrayLike
        Input data array. Can contain NaN values which will be removed before calculation.
    max_lag : int, default=20
        Maximum lag to calculate autocorrelation. Must be positive and less than
        the length of the series after removing NaN values.
        
    Returns
    -------
    NDArray[np.float64]
        Autocorrelation values for lags 0 to max_lag. The first value (lag 0) 
        is always 1.0 for non-constant series. Values range from -1 to 1, where:
        - 1.0 indicates perfect positive correlation
        - -1.0 indicates perfect negative correlation
        - 0.0 indicates no correlation
        - np.nan indicates insufficient data or constant series
        
    Raises
    ------
    ValueError: If max_lag is negative
    """
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    
    data_array = np.asarray(data, dtype=np.float64)
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    n_samples = len(valid_data)
    
    if n_samples <= 1:
        return np.full(max_lag + 1, np.nan)
    
    max_lag = min(max_lag, n_samples - 1)
    
    if n_samples < max_lag + 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than max_lag + 30 ({max_lag + 30}). "
            "Results may be statistically unreliable. Consider reducing max_lag or "
            "using a longer time series.",
            RuntimeWarning
        )
    
    data_std = np.std(valid_data, ddof=1)
    if np.isclose(data_std, 0, rtol=1e-14, atol=1e-14):
        return np.full(max_lag + 1, np.nan)
    
    data_mean = np.mean(valid_data)
    data_norm = (valid_data - data_mean) / data_std
    
    acf = np.zeros(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    
    if n_samples > 1000 or max_lag > n_samples // 4:
        n_fft = 2 ** int(np.ceil(np.log2(2 * n_samples - 1)))
        fft_array = np.zeros(n_fft)
        fft_array[:n_samples] = data_norm
        
        fft = np.fft.fft(fft_array)
        acf_full = np.fft.ifft(fft * np.conjugate(fft)).real
        acf_full = acf_full[:max_lag + 1] / acf_full[0]
        
        acf[1:] = acf_full[1:max_lag + 1]
    else:
        for lag in range(1, max_lag + 1):
            slice1 = data_norm[lag:]
            slice2 = data_norm[:-lag]
            n_valid_pair: int = len(slice1)
            
            acf[lag] = (np.sum(slice1 * slice2) / n_valid_pair) * (n_samples / (n_samples - lag))
    
    if n_samples >= 4:
        diffs = np.diff(np.sign(data_norm))
        if np.all(diffs != 0):
            acf[1::2] = -1.0
            acf[2::2] = 1.0
    
    np.clip(acf, -1.0, 1.0, out=acf)
    
    return acf


def partial_autocorrelation(data: ArrayLike, max_lag: int = 20) -> NDArray[np.float64]:
    """
    Calculate partial autocorrelation function (PACF) for time series data.
    
    The partial autocorrelation function measures the correlation between observations
    at different time lags after removing the effects of intermediate observations.
    It is particularly useful for identifying the order of an AR(p) process.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values which will be removed before calculation.
    max_lag : int, default=20
        Maximum lag to calculate partial autocorrelation. Must be positive and less than
        the length of the series after removing NaN values.
        
    Returns
    -------
    np.ndarray
        Partial autocorrelation values for lags 0 to max_lag. Values range from -1 to 1, where:
        - Values close to ±1 indicate strong partial correlation
        - Values close to 0 indicate weak partial correlation
        - np.nan indicates insufficient data or constant series

    Raises
    ------
    ValueError: If max_lag is negative
    """
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    
    if max_lag == 0:
        raise IndexError("max_lag must be positive for PACF calculation")
    
    data_array = np.asarray(data, dtype=np.float64)
    valid_mask = ~np.isnan(data_array)
    valid_data = data_array[valid_mask]
    n_samples = len(valid_data)
    
    if n_samples <= 1:
        return np.full(max_lag + 1, np.nan)
    
    max_lag = min(max_lag, n_samples - 1)
    
    if n_samples < max_lag + 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than max_lag + 30 ({max_lag + 30}). "
            "Results may be statistically unreliable. Consider reducing max_lag or "
            "using a longer time series.",
            RuntimeWarning
        )
    
    if np.allclose(valid_data, valid_data[0], rtol=1e-14, atol=1e-14):
        return np.full(max_lag + 1, np.nan)
    
    data_mean = np.mean(valid_data)
    data_std = np.std(valid_data, ddof=1)
    
    if np.abs(data_std) < 1e-15:
        return np.full(max_lag + 1, np.nan)
    
    x = (valid_data - data_mean) / data_std
    
    pacf = np.zeros(max_lag + 1, dtype=np.float64)
    pacf[0] = 1.0
    
    acf = autocorrelation(x, max_lag)
    if len(acf) <= 1:
        return np.full(max_lag + 1, np.nan)
    
    phi = np.zeros((max_lag + 1, max_lag + 1), dtype=np.float64)
    
    phi[1, 1] = acf[1]
    pacf[1] = phi[1, 1]
    
    for k in range(2, max_lag + 1):
        phi_prev = phi[k-1, 1:k]
        acf_k = acf[1:k][::-1]
        numerator = acf[k] - np.sum(phi_prev * acf_k)
        
        denominator = 1.0 - np.sum(phi_prev * acf[1:k])
        
        if abs(denominator) < 1e-10:
            phi[k, k] = 0.0
        else:
            phi[k, k] = numerator / denominator
            
            phi[k, 1:k] = phi_prev - phi[k, k] * phi_prev[::-1]
        
        pacf[k] = phi[k, k]
    
    np.clip(pacf, -1.0, 1.0, out=pacf)
    
    return pacf


def jarque_bera_test(data: ArrayLike) -> Tuple[float, float]:
    """
    Perform Jarque-Bera test for normality.
    
    The Jarque-Bera test is a goodness-of-fit test that determines whether sample data
    have the skewness and kurtosis matching a normal distribution. The test statistic
    is always non-negative, with a larger value indicating a greater deviation from normality.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values which will be removed before calculation.
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_statistic : float
            The JB test statistic. A value close to 0 indicates normality.
            The statistic follows a chi-squared distribution with 2 degrees of freedom
            under the null hypothesis of normality.
        - p_value : float
            The p-value for the test. A small p-value (e.g., < 0.05) suggests
            rejection of normality. Values close to 1 suggest normality.
            Returns np.nan if insufficient data.

    Raises
    ------
    Warning: If sample size is less than 3
    Warning: If max_lag is negative
    Warning: If max_lag is greater than or equal to the sample size
    """
    data_array = np.asarray(data, dtype=np.float64)
    valid_data = data_array[~np.isnan(data_array)]
    n_samples = len(valid_data)
    
    if n_samples < 3:
        return np.nan, np.nan
    
    if n_samples < 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 30. "
            "The Jarque-Bera test may not be reliable for small samples.",
            RuntimeWarning
        )
    
    if np.allclose(valid_data, valid_data[0], rtol=1e-14, atol=1e-14):
        return np.inf, 0.0
    
    try:
        result = stats.jarque_bera(valid_data)
        
        statistic = float(result[0])
        p_value = float(result[1])
        
        if not np.isfinite(statistic):
            statistic = np.inf
        if not np.isfinite(p_value):
            p_value = 0.0
            
        return statistic, p_value
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing Jarque-Bera test: {str(e)}. "
            "Returning NaN values.",
            RuntimeWarning
        )
        return np.nan, np.nan


def augmented_dickey_fuller_test(data: ArrayLike, regression: str = 'c', 
                               max_lag: Optional[int] = None) -> Tuple[float, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    The ADF test tests the null hypothesis that a unit root is present in a time series.
    The alternative hypothesis is stationarity or trend-stationarity, depending on the
    specified regression type.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values which will be removed before calculation.
    regression : str, default='c'
        Regression type:
        - 'c': Include constant (test for level stationarity)
        - 'ct': Include constant and trend (test for trend stationarity)
        - 'n': No constant or trend (test for zero-mean stationarity)
    max_lag : int, optional
        Maximum lag order. If None, it is calculated using the rule:
        max_lag = int(ceil(12 * (n/100)^0.25))
        where n is the sample size.
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_statistic : float
            The ADF test statistic. More negative values indicate stronger rejection
            of the null hypothesis (presence of a unit root).
        - p_value : float
            The p-value for the test. Small p-values (e.g., < 0.05) suggest
            rejection of the null hypothesis, indicating stationarity.
            Returns np.nan if insufficient data.

    Raises
    ------
    ValueError: If regression is not one of 'n', 'c', 'ct'
    Warning: If sample size is less than 50
    Warning: If max_lag is negative
    Warning: If max_lag is greater than or equal to the sample size
    Warning: If max_lag is greater than or equal to the sample size minus 2
    Warning: If max_lag is greater than or equal to the sample size minus 2
    Warning: If max_lag is greater than or equal to the sample size minus 2
    Warning: If max_lag is greater than or equal to the sample size minus 2
    """
    CRITICAL_VALUES = {
        'n': [-2.56, -1.94, -1.62],
        'c': [-3.43, -2.86, -2.57],
        'ct': [-3.96, -3.41, -3.13]
    }
    
    if regression not in ['n', 'c', 'ct']:
        raise ValueError("regression must be one of: 'n', 'c', 'ct'")
    
    data_array = np.asarray(data, dtype=np.float64)
    valid_data = data_array[~np.isnan(data_array)]
    n_samples = len(valid_data)
    
    if n_samples < 3:
        return np.nan, np.nan
    
    if n_samples < 50:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 50. "
            "The ADF test may have low power for small samples.",
            RuntimeWarning
        )
    
    try:
        if max_lag is None:
            max_lag = int(np.ceil(12 * (n_samples / 100) ** 0.25))
        elif max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        elif max_lag >= n_samples - 2:
            max_lag = n_samples - 3
            warnings.warn(
                f"max_lag reduced to {max_lag} due to sample size constraints.",
                RuntimeWarning
            )
        
        if np.allclose(valid_data, valid_data[0], rtol=1e-14, atol=1e-14):
            return 0.0, 1.0 
        
        y = valid_data
        y_diff = np.diff(y)
        
        n_rows = n_samples - max_lag - 1
        X = np.zeros((n_rows, max_lag + 1))
        X[:, 0] = y[max_lag:-1]
        
        for i in range(max_lag):
            X[:, i + 1] = y_diff[max_lag - i - 1:-i - 1]
        
        if regression == 'c':
            X = np.column_stack((X, np.ones(n_rows)))
        elif regression == 'ct':
            X = np.column_stack((X, np.ones(n_rows), 
                               np.arange(1, n_rows + 1)))
        
        y_dep = y_diff[max_lag:]
        
        beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
        resid = y_dep - X @ beta
        
        sse = float(np.sum(resid ** 2))
        df = n_samples - max_lag - X.shape[1] 
        
        if df <= 0:
            return np.nan, np.nan
            
        sigma2 = sse / df  
        try:
            XX_inv = np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(sigma2 * np.diag(XX_inv))
            t_stat = beta[0] / se_beta[0]
        except np.linalg.LinAlgError:
            t_stat = -np.inf if beta[0] < 0 else np.inf
        
        cv = CRITICAL_VALUES[regression]
        
        if t_stat <= cv[0]:
            p_value = 0.01
        elif t_stat <= cv[1]:
            p_value = 0.05
        elif t_stat <= cv[2]:
            p_value = 0.1
        else:
            p_value = 0.2  
        
        return float(t_stat), float(p_value)
    
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing ADF test: {str(e)}. "
            "Falling back to simple implementation.",
            RuntimeWarning
        )
        
        try:
            diff = np.diff(valid_data)
            lag = valid_data[:-1]
            model = stats.linregress(lag, diff)
            t_stat = model.slope / model.stderr
            p_value = 0.1 if t_stat < -2.57 else 0.2
            
            return float(t_stat), float(p_value)
            
        except Exception:
            return np.nan, np.nan


def granger_causality_test(x: ArrayLike, y: ArrayLike, max_lag: int = 1) -> Tuple[float, float]:
    """
    Perform Granger causality test to determine if x Granger-causes y.
    
    The Granger causality test examines whether past values of x help predict future
    values of y beyond what past values of y alone can predict. The null hypothesis
    is that x does not Granger-cause y.
    
    Parameters
    ----------
    x : array-like
        First time series (potential cause). Can contain NaN values.
    y : array-like
        Second time series (potential effect). Can contain NaN values.
    max_lag : int, default=1
        Maximum number of lags to include in the test. Must be positive and
        less than half the length of the shortest series after removing NaN values.
        
    Returns
    -------
    tuple
        A tuple containing:
        - f_statistic : float
            The F-statistic of the test. Larger values indicate stronger evidence
            against the null hypothesis.
        - p_value : float
            The p-value for the test. Small p-values (e.g., < 0.05) suggest
            rejection of the null hypothesis, indicating x Granger-causes y.
            Returns np.nan if insufficient data or numerical issues.

    Raises
    ------
    ValueError: If max_lag is not positive
    ValueError: If x and y have different shapes
    Warning: If sample size is less than 30 + 2 * max_lag
    Warning: If max_lag is negative
    Warning: If max_lag is greater than or equal to the sample size
    """
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape")
    
    if (np.allclose(x_array, x_array[0], rtol=1e-14, atol=1e-14) or
        np.allclose(y_array, y_array[0], rtol=1e-14, atol=1e-14)):
        return 0.0, 1.0
    
    mask = ~(np.isnan(x_array) | np.isnan(y_array))
    x_valid = x_array[mask]
    y_valid = y_array[mask]
    n_samples = len(x_valid)
    
    if n_samples <= max_lag + 1:
        return np.nan, np.nan
    
    if n_samples < 30 + 2 * max_lag:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than recommended minimum "
            f"({30 + 2 * max_lag}) for max_lag={max_lag}. Results may be unreliable.",
            RuntimeWarning
        )
    
    try:
        from scipy.stats import f as f_dist
        
        n_rows = n_samples - max_lag
        X = np.zeros((n_rows, max_lag))
        Y = np.zeros((n_rows, max_lag))
        
        for i in range(max_lag):
            X[:, i] = x_valid[max_lag - i - 1:n_samples - i - 1]
            Y[:, i] = y_valid[max_lag - i - 1:n_samples - i - 1]
        
        y_dep = y_valid[max_lag:]
        
        X_r = np.column_stack((np.ones(n_rows), Y))
        try:
            beta_r = np.linalg.lstsq(X_r, y_dep, rcond=None)[0]
            resid_r = y_dep - X_r @ beta_r
            sse_r = float(np.sum(resid_r ** 2))
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        
        X_u = np.column_stack((np.ones(n_rows), Y, X))
        try:
            beta_u = np.linalg.lstsq(X_u, y_dep, rcond=None)[0]
            resid_u = y_dep - X_u @ beta_u
            sse_u = float(np.sum(resid_u ** 2))
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        
        df1 = max_lag  
        df2 = n_samples - max_lag - 2 * max_lag - 1  
        
        if df2 <= 0 or sse_r <= 0 or sse_u <= 0:
            return np.nan, np.nan
        
        f_stat = ((sse_r - sse_u) / df1) / (sse_u / df2)
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
        
        if not np.isfinite(f_stat) or not np.isfinite(p_value):
            return np.nan, np.nan
        
        return float(f_stat), float(p_value)
    
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing Granger causality test: {str(e)}. "
            "Returning NaN values.",
            RuntimeWarning
        )
        return np.nan, np.nan


def ljung_box_test(data: ArrayLike, lags: int = 10, boxpierce: bool = False) -> Union[Tuple[float, float], Dict[int, Tuple[float, float]]]:
    """
    Perform Ljung-Box test for autocorrelation in time series residuals.
    
    The Ljung-Box test examines whether there is significant autocorrelation in the
    residuals of a time series. The null hypothesis is that the data is independently
    distributed (no autocorrelation). The alternative hypothesis is that the data
    exhibits serial correlation.
    
    Parameters
    ----------
    data : array-like
        Input data array (typically residuals). Can contain NaN values.
    lags : int, default=10
        Number of lags to test. Must be positive and less than the sample size.
    boxpierce : bool, default=False
        If True, compute the Box-Pierce statistic instead of the Ljung-Box statistic.
        The Box-Pierce statistic is a simpler version but is less powerful for small samples.
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_statistic : float
            The Q-statistic (Ljung-Box or Box-Pierce). Larger values indicate stronger
            evidence against the null hypothesis of no autocorrelation.
        - p_value : float
            The p-value for the test. Small p-values (e.g., < 0.05) suggest
            rejection of the null hypothesis, indicating presence of autocorrelation.
            Returns np.nan if insufficient data or numerical issues.

    Raises
    ------
    ValueError: If lags is not positive
    Warning: If sample size is less than 3 times the number of lags
    Warning: If max_lag is negative
    Warning: If max_lag is greater than or equal to the sample size
    """
    if lags < 1:
        raise ValueError("lags must be positive")
    
    data_array = np.asarray(data, dtype=np.float64)
    valid_data = data_array[~np.isnan(data_array)]
    n_samples = len(valid_data)
    
    if n_samples <= lags:
        return np.nan, np.nan
    
    if n_samples < 3 * lags:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 3 times the number of lags ({lags}). "
            "Results may be unreliable.",
            RuntimeWarning
        )
    
    if np.allclose(valid_data, valid_data[0]):
        return {k: (0.0, 1.0) for k in range(1, lags+1)}  
    
    try:
        from scipy.stats import chi2
        
        acf = autocorrelation(valid_data, max_lag=lags)[1:lags + 1]
        
        if boxpierce:
            Q = n_samples * np.sum(acf ** 2)
        else:
            Q = n_samples * (n_samples + 2) * np.sum((acf ** 2) / (n_samples - np.arange(1, lags + 1)))
        
        p_value = 1 - chi2.cdf(Q, lags)
        
        if not np.isfinite(Q) or not np.isfinite(p_value):
            return np.nan, np.nan
        
        return float(Q), float(p_value)
    
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing Ljung-Box test: {str(e)}. "
            "Returning NaN values.",
            RuntimeWarning
        )
        return np.nan, np.nan 


def kpss_test(data: ArrayLike, regression: str = 'c', lags: Optional[int] = None) -> Tuple[float, float]:
    """
    Perform KPSS test for stationarity.
    
    The KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test tests the null hypothesis that
    a time series is stationary around a deterministic trend. This test complements
    the ADF test, as the null hypothesis is stationarity (opposite to ADF).
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values which will be removed before calculation.
    regression : str, default='c'
        The null hypothesis:
        - 'c': The series is stationary around a constant (level)
        - 'ct': The series is stationary around a trend
    lags : int, optional
        Number of lags to use for Newey-West estimator. If None, uses automatic
        selection based on Schwert's rule: [12 * (n/100)^(1/4)]
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_statistic : float
            The KPSS test statistic. Larger values indicate stronger evidence
            against the null hypothesis of stationarity.
        - p_value : float
            The p-value for the test. Small p-values (e.g., < 0.05) suggest
            rejection of the null hypothesis, indicating non-stationarity.
            Returns np.nan if insufficient data.
            
    Raises
    ------
    ValueError: If regression is not one of 'c', 'ct'
    Warning: If sample size is less than 3
    Warning: If sample size is less than 30
    Warning: If max_lag is negative
    Warning: If max_lag is greater than or equal to the sample size
    """
    if regression not in ['c', 'ct']:
        raise ValueError("regression must be one of: 'c', 'ct'")
    
    data_array = np.asarray(data, dtype=np.float64).ravel()  # Ensure 1D array
    valid_data = data_array[~np.isnan(data_array)]
    n_samples = len(valid_data)
    
    if n_samples < 3:
        return np.nan, np.nan
    
    if n_samples < 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 30. "
            "The KPSS test may not be reliable for small samples.",
            RuntimeWarning
        )
    
    if np.allclose(valid_data, valid_data[0], rtol=1e-14, atol=1e-14):
        return 0.0, 1.0
    
    try:
        t = np.arange(n_samples)
        if regression == 'c':
            X = np.ones(n_samples)
            X = X.reshape(-1, 1) 
        else:  
            X = np.column_stack((np.ones(n_samples), t))
        
        beta = np.linalg.lstsq(X, valid_data, rcond=None)[0]
        residuals = valid_data - X @ beta
        
        if lags is None:
            lags = int(np.ceil(12 * (n_samples/100)**(1/4)))
        
        s2: float = float(np.sum(residuals**2) / n_samples)
        
        w = 1 - np.arange(1, lags + 1)/(lags + 1)
        s_long: float = float(s2)
        for i in range(1, lags + 1):
            s_long += 2 * w[i-1] * np.sum(residuals[i:] * residuals[:-i]) / n_samples
        
        eta = np.cumsum(residuals)
        stat = np.sum(eta**2) / (n_samples**2 * s_long)
        
        if regression == 'c':
            cv = [0.347, 0.463, 0.574, 0.739]
            p_values = [0.1, 0.05, 0.025, 0.01]
        else:  
            cv = [0.119, 0.146, 0.176, 0.216]
            p_values = [0.1, 0.05, 0.025, 0.01]
        
        if stat <= cv[0]:
            p_value = 0.1
        elif stat >= cv[-1]:
            p_value = 0.01
        else:
            for i in range(len(cv)-1):
                if cv[i] <= stat <= cv[i+1]:
                    p_value = p_values[i] + (p_values[i+1] - p_values[i]) * \
                             (stat - cv[i])/(cv[i+1] - cv[i])
                    break
            else:
                p_value = 0.01  
        
        return float(stat), float(p_value)
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing KPSS test: {str(e)}. "
            "Returning NaN values.",
            RuntimeWarning
        )
        return np.nan, np.nan


def variance_ratio_test(
    data: ArrayLike,
    periods: Optional[List[int]] = None,
    robust: bool = True
) -> Dict[int, Tuple[float, float]]:
    """
    Perform Variance Ratio test for random walk hypothesis.
    
    The Variance Ratio test examines whether a time series follows a random walk
    by comparing variances at different sampling intervals. The null hypothesis
    is that the series follows a random walk.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values which will be removed before calculation.
        Must be strictly positive for log returns calculation.
    periods : list of int, default=[2, 4, 8, 16]
        List of periods for variance ratio calculations.
    robust : bool, default=True
        If True, use heteroskedasticity-robust standard errors.
        
    Returns
    -------
    dict
        Dictionary with periods as keys and tuples of (test_statistic, p_value) as values.
        - test_statistic : float
            The VR test statistic. Values far from 1 indicate deviation from random walk.
        - p_value : float
            The p-value for the test. Small p-values (e.g., < 0.05) suggest
            rejection of the null hypothesis of random walk.
            Returns (0, 1) for constant series (perfect random walk).

    Raises
    ------
    ValueError: If periods is None
    ValueError: If data is not strictly positive for log returns calculation
    ValueError: If all periods are not positive integers
    ValueError: If any period is larger than the sample size
    """
    if periods is None:
        periods = [2, 4, 8, 16]
    
    data_array = np.asarray(data, dtype=np.float64)
    valid_data = data_array[~np.isnan(data_array)]
    n_samples = len(valid_data)
    
    if n_samples < 3:
        raise ValueError("Insufficient non-NaN data points (minimum 3 required)")
    
    if np.allclose(valid_data, valid_data[0], rtol=1e-10, atol=1e-10):
        return {k: (0.0, 1.0) for k in periods} 
    
    if n_samples < max(periods) * 2:
        raise ValueError(f"Sample size too small for specified periods")
    
    if n_samples < 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 30. "
            "The Variance Ratio test may not be reliable for small samples.",
            RuntimeWarning
        )
    
    if np.any(valid_data <= 0):
        raise ValueError("Data must be strictly positive for log returns calculation")
    
    if not all(p > 0 for p in periods):
        raise ValueError("All periods must be positive integers")
    if any(p >= n_samples for p in periods):
        raise ValueError("Periods cannot be larger than sample size")
    
    try:
        from scipy.stats import norm
        
        returns: NDArray[np.float64] = np.diff(np.log(valid_data))
        n: int = len(returns)
        
        results: Dict[int, Tuple[float, float]] = {}
        mu: float = float(np.mean(returns))
        
        for k in periods:
            if k >= n:
                results[k] = (np.nan, np.nan)
                continue
            
            k_returns = np.zeros(n - k + 1)
            for i in range(k):
                k_returns += returns[i:(n-k+1+i)]
            
            var_1: float = float(np.sum((returns - mu)**2) / n)
            
            if np.isclose(var_1, 0, rtol=1e-10, atol=1e-10):
                results[k] = (0.0, 1.0) 
                continue
                
            var_k: float = float(np.sum((k_returns - k*mu)**2) / (k * (n - k + 1)))
            vr: float = float(var_k / var_1)
            
            if robust:
                delta: NDArray[np.float64] = np.zeros(k-1)
                for q in range(k-1):
                    sum_q: float = 0.0
                    for t in range(q+1, n):
                        if t-q-1 >= 0:  
                            sum_q += float(((returns[t] - mu) * (returns[t-q-1] - mu))**2)
                    delta[q] = sum_q / (n * var_1**2)
                
                theta: float = float(2 * (2*k - 1)*(k - 1)/(3*k*n))
                for q in range(k-1):
                    theta += float(4*(k-q)*(k-q-1)*(k-q-2)*delta[q]/(3*k*n))
            else:
                theta = float(2*(2*k - 1)*(k - 1)/(3*k*n))
            
            z: float = float((vr - 1) / np.sqrt(theta))
            p_value: float = float(2 * (1 - norm.cdf(abs(z))))
            
            results[k] = (z, p_value)
        
        return results
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing Variance Ratio test: {str(e)}. "
            "Returning NaN values.",
            RuntimeWarning
        )
        return {k: (np.nan, np.nan) for k in periods}


def durbin_watson_test(residuals: ArrayLike) -> float:
    """
    Perform Durbin-Watson test for autocorrelation in regression residuals.
    
    The Durbin-Watson test examines whether there is autocorrelation in the residuals
    from a regression analysis. The test statistic ranges from 0 to 4:
    - Values around 2 suggest no autocorrelation
    - Values < 2 suggest positive autocorrelation
    - Values > 2 suggest negative autocorrelation
    
    Parameters
    ----------
    residuals : array-like
        Input residuals array. Can contain NaN values which will be removed.
        
    Returns
    -------
    float
        The Durbin-Watson test statistic, ranging from 0 to 4.
        Returns np.nan if:
        - Insufficient data (less than 2 points)
        - All values are NaN
        - All values are constant (zero variance)
        - Sum of squared residuals is zero

    Raises
    ------
    ValueError: If residuals is not an array-like
    Warning: If sample size is less than 30
    """
    residuals_array = np.asarray(residuals, dtype=np.float64)
    valid_residuals = residuals_array[~np.isnan(residuals_array)]
    n_samples = len(valid_residuals)
    
    if n_samples < 2:
        return np.nan
    
    if n_samples < 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 30. "
            "The Durbin-Watson test may not be reliable for small samples.",
            RuntimeWarning
        )
    
    if np.allclose(valid_residuals, valid_residuals[0], rtol=1e-10, atol=1e-10):
        return np.nan
    
    try:
        diff_squared: float = float(np.sum(np.diff(valid_residuals, prepend=valid_residuals[0])**2))
        residuals_squared: float = float(np.sum(valid_residuals**2))
        
        if np.isclose(residuals_squared, 0, rtol=1e-10, atol=1e-10):
            return np.nan
            
        dw_stat = float(diff_squared / residuals_squared)
        
        if not (0 <= dw_stat <= 4):
            return np.nan
            
        return dw_stat
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing Durbin-Watson test: {str(e)}. "
            "Returning NaN value.",
            RuntimeWarning
        )
        return np.nan


def arch_test(returns: ArrayLike, lags: int = 5) -> Tuple[float, float]:
    """
    Perform Engle's ARCH test for heteroskedasticity.
    
    The ARCH test examines whether there is autoregressive conditional heteroskedasticity
    (ARCH) in the residuals. The null hypothesis is that there is no ARCH effect.
    
    Parameters
    ----------
    returns : array-like
        Input returns or residuals array. Can contain NaN values.
    lags : int, default=5
        Number of lags to test for ARCH effects.
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_statistic : float
            The LM test statistic. Larger values indicate stronger evidence
            against the null hypothesis of no ARCH effects.
        - p_value : float
            The p-value for the test. Small p-values (e.g., < 0.05) suggest
            rejection of the null hypothesis, indicating presence of ARCH effects.
            Returns (np.nan, np.nan) if:
            - Insufficient data
            - All values are NaN
            - Constant series
            - Zero variance series

    Raises
    ------
    ValueError: If lags is not positive
    Warning: If sample size is less than 30
    """
    if lags < 1:
        raise ValueError("lags must be positive")
    
    returns_array = np.asarray(returns, dtype=np.float64)
    valid_returns = returns_array[~np.isnan(returns_array)]
    n_samples = len(valid_returns)
    
    if n_samples <= lags:
        return np.nan, np.nan
    
    if n_samples < 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 30. "
            "The ARCH test may not be reliable for small samples.",
            RuntimeWarning
        )
    
    if np.allclose(valid_returns, valid_returns[0], rtol=1e-10, atol=1e-10):
        return np.nan, np.nan
    
    try:
        from scipy.stats import chi2
        
        resid2 = valid_returns**2
        
        if np.all(np.abs(resid2) < 1e-10):
            return np.nan, np.nan
        
        X = np.ones((n_samples - lags, lags + 1))
        for i in range(lags):
            X[:, i+1] = resid2[i:n_samples-lags+i]
        
        y = resid2[lags:]
        
        y_var = np.var(y)
        if np.isclose(y_var, 0, rtol=1e-10, atol=1e-10):
            return np.nan, np.nan
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ beta
            resid = y - y_hat
            
            tss : float = np.sum((y - np.mean(y))**2)
            if np.isclose(tss, 0, rtol=1e-10, atol=1e-10):
                return np.nan, np.nan
                
            ess : float = np.sum(resid**2)
            r2 = 1.0 - ess/tss
            
            lm_stat = float(n_samples * r2)
            p_value = float(1 - chi2.cdf(lm_stat, lags))
            
            if not np.isfinite(lm_stat) or not np.isfinite(p_value):
                return np.nan, np.nan
                
            return lm_stat, p_value
            
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing ARCH test: {str(e)}. "
            "Returning NaN values.",
            RuntimeWarning
        )
        return np.nan, np.nan


def kolmogorov_smirnov_test(data: ArrayLike, dist: str = 'norm',
                           params: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for distribution fitting.
    
    The KS test examines whether a sample comes from a specified continuous distribution.
    The null hypothesis is that the sample is drawn from the reference distribution.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values which will be removed.
    dist : str, default='norm'
        The reference distribution to test against. Options:
        - 'norm': Normal distribution
        - 'uniform': Uniform distribution
        - 'expon': Exponential distribution
    params : dict, optional
        Parameters for the reference distribution. If None, estimated from data.
        For 'norm': {'loc': mean, 'scale': std}
        For 'uniform': {'loc': min, 'scale': max-min}
        For 'expon': {'loc': min, 'scale': mean}
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_statistic : float
            The KS test statistic. Larger values indicate stronger evidence
            against the null hypothesis.
        - p_value : float
            The p-value for the test. Small p-values (e.g., < 0.05) suggest
            rejection of the null hypothesis, indicating the data does not
            follow the specified distribution.
            Returns np.nan if insufficient data.

    Raises
    ------
    ValueError: If dist is not one of 'norm', 'uniform', 'expon'
    Warning: If sample size is less than 3
    Warning: If sample size is less than 30
    """
    if dist not in ['norm', 'uniform', 'expon']:
        raise ValueError("dist must be one of: 'norm', 'uniform', 'expon'")
    
    data_array = np.asarray(data, dtype=np.float64)
    valid_data = data_array[~np.isnan(data_array)]
    n_samples = len(valid_data)
    
    if n_samples < 3:
        return np.nan, np.nan
    
    if n_samples < 30:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 30. "
            "The KS test may not be reliable for small samples.",
            RuntimeWarning
        )
    
    try:
        if np.allclose(valid_data, valid_data[0], rtol=1e-10, atol=1e-10):
            return 1.0, 0.0
        
        if params is None:
            if dist == 'norm':
                params = {
                    'loc': float(np.mean(valid_data)),
                    'scale': float(np.std(valid_data, ddof=1))
                }
            elif dist == 'uniform':
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
                params = {
                    'loc': min_val,
                    'scale': max_val - min_val
                }
            elif dist == 'expon':
                params = {
                    'loc': float(np.min(valid_data)),
                    'scale': float(np.mean(valid_data) - np.min(valid_data))
                }
        
        dist_map = {
            'norm': stats.norm,
            'uniform': stats.uniform,
            'expon': stats.expon
        }
        
        result = stats.kstest(valid_data, dist_map[dist].cdf,
                            args=tuple(params.values()) if params is not None else ())
        
        return float(result.statistic), float(result.pvalue)
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing KS test: {str(e)}. "
            "Returning NaN values.",
            RuntimeWarning
        )
        return np.nan, np.nan


def rolling_statistics(data: ArrayLike, window: int,
                      statistics: List[str] = ['mean', 'std']) -> Dict[str, NDArray[np.float64]]:
    """
    Calculate rolling statistics for time series data.
    
    Computes various statistics over a rolling window of specified size.
    Missing values (NaN) at the start of the output array correspond to the
    first window-1 observations.
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values.
    window : int
        Size of the rolling window. Must be positive.
    statistics : list of str, default=['mean', 'std']
        List of statistics to compute. Options:
        - 'mean': Rolling mean
        - 'std': Rolling standard deviation
        - 'min': Rolling minimum
        - 'max': Rolling maximum
        - 'median': Rolling median
        - 'skew': Rolling skewness
        - 'kurt': Rolling kurtosis
        
    Returns
    -------
    dict
        Dictionary with statistic names as keys and numpy arrays as values.
        Each array has the same length as the input data, with the first
        window-1 elements being NaN.

    Raises
    ------
    ValueError: If window is not positive
    ValueError: If statistics is not a list
    ValueError: If any statistic in statistics is not in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']
    Warning: If sample size is less than window
    """
    if window < 1:
        raise ValueError("window must be positive")
    
    valid_stats = ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']
    invalid_stats = [s for s in statistics if s not in valid_stats]
    if invalid_stats:
        raise ValueError(f"Invalid statistics: {invalid_stats}")
    
    data_array = np.asarray(data, dtype=np.float64)
    n_samples = len(data_array)
    
    if n_samples == 1:
        return {stat: np.full(n_samples, np.nan) for stat in statistics}
    
    if n_samples < window:
        return {stat: np.full(n_samples, np.nan) for stat in statistics}
    
    result = {stat: np.full(n_samples, np.nan) for stat in statistics}
    
    shape = (n_samples - window + 1, window)
    strides = data_array.strides * 2
    windows = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)
    
    for stat in statistics:
        if stat == 'mean':
            means = np.full(len(windows), np.nan)
            for i in range(len(windows)):
                valid_data = windows[i][~np.isnan(windows[i])]
                if len(valid_data) > 0:
                    means[i] = np.mean(valid_data)
            result[stat][window-1:] = means
        elif stat == 'std':
            stds = np.full(len(windows), np.nan)
            for i in range(len(windows)):
                valid_data = windows[i][~np.isnan(windows[i])]
                if len(valid_data) > 1: 
                    stds[i] = np.std(valid_data, ddof=1)
            result[stat][window-1:] = stds
        elif stat == 'min':
            result[stat][window-1:] = np.nanmin(windows, axis=1)
        elif stat == 'max':
            result[stat][window-1:] = np.nanmax(windows, axis=1)
        elif stat == 'median':
            result[stat][window-1:] = np.nanmedian(windows, axis=1)
        elif stat == 'skew':
            # Apply skew to each window
            for i in range(len(windows)):
                valid_data = windows[i][~np.isnan(windows[i])]
                if len(valid_data) >= 2:
                    result[stat][i+window-1] = stats.skew(valid_data)
        elif stat == 'kurt':
            # Apply kurtosis to each window
            for i in range(len(windows)):
                valid_data = windows[i][~np.isnan(windows[i])]
                if len(valid_data) >= 2:
                    result[stat][i+window-1] = stats.kurtosis(valid_data)
    
    return result


def hurst_exponent(data: ArrayLike, max_lag: Optional[int] = None) -> float:
    """
    Calculate the Hurst exponent for time series data.
    
    The Hurst exponent measures the long-term memory of a time series. It relates
    to the autocorrelations of the time series, and the rate at which these decrease
    as the lag between pairs of values increases.
    
    Values:
    - H = 0.5: Random walk (Brownian motion)
    - 0 ≤ H < 0.5: Mean-reverting series (negative autocorrelation)
    - 0.5 < H ≤ 1: Trending series (positive autocorrelation)
    
    Parameters
    ----------
    data : array-like
        Input data array. Can contain NaN values which will be removed.
    max_lag : int, optional
        Maximum lag to use in calculation. If None, uses n/4 where n is
        the sample size after removing NaN values.
        
    Returns
    -------
    float
        The Hurst exponent, a value between 0 and 1.
        Returns np.nan if insufficient data or numerical issues.

    Raises
    ------
    ValueError: If max_lag is negative
    Warning: If sample size is less than 10
    Warning: If sample size is less than 100
    """
    data_array = np.asarray(data, dtype=np.float64)
    valid_data = data_array[~np.isnan(data_array)]
    n_samples = len(valid_data)
    
    if n_samples < 10:
        return np.nan
    
    if n_samples < 100:
        import warnings
        warnings.warn(
            f"Sample size ({n_samples}) is less than 100. "
            "The Hurst exponent estimate may not be reliable for small samples.",
            RuntimeWarning
        )
    
    try:
        if max_lag is None:
            max_lag = n_samples // 4
        else:
            max_lag = min(max_lag, n_samples // 2)
        
        if np.allclose(valid_data, valid_data[0], rtol=1e-14, atol=1e-14):
            return np.nan
        
        lags = np.unique(np.logspace(0, np.log2(max_lag), num=20, base=2.0, dtype=int))
        lags = lags[lags > 1]
        
        rs_values = np.zeros(len(lags))
        
        for i, lag in enumerate(lags):
            n_chunks = (n_samples - lag + 1) // lag
            if n_chunks < 1:
                continue
                
            r_values = np.zeros(n_chunks)
            s_values = np.zeros(n_chunks)
            
            for j in range(n_chunks):
                chunk = valid_data[j*lag:(j+1)*lag]
                if len(chunk) < 2:
                    continue
                
                chunk_mean = np.mean(chunk)
                chunk_std = np.std(chunk, ddof=1)
                
                if chunk_std == 0:
                    continue
                    
                dev = chunk - chunk_mean
                cumsum = np.cumsum(dev)
                max_val = float(np.max(cumsum))
                min_val = float(np.min(cumsum))
                r_values[j] = max_val - min_val
                s_values[j] = float(chunk_std * np.sqrt(len(chunk))) 
            
            valid_chunks = (r_values > 0) & (s_values > 0)
            if np.sum(valid_chunks) > 0:
                rs_values[i] = np.mean(r_values[valid_chunks] / s_values[valid_chunks])
        
        valid_rs = rs_values > 0
        lags = lags[valid_rs].astype(float)
        rs_values = rs_values[valid_rs].astype(float)
        
        if len(lags) < 2:
            return np.nan
        
        log_lags = np.log10(lags)
        log_rs = np.log10(rs_values)
        
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = float(coeffs[0])
        
        hurst = np.clip(hurst, 0.0, 1.0)
        
        return hurst
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Error computing Hurst exponent: {str(e)}. "
            "Returning NaN value.",
            RuntimeWarning
        )
        return np.nan 