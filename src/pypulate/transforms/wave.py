"""
Wave and zigzag transforms for financial time series data.

This module provides functions for extracting wave points and zigzag patterns
from financial time series data, which are useful for technical analysis.
"""

import numpy as np
from typing import Union, List
from numpy.typing import NDArray, ArrayLike
def wave(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike
) -> NDArray[np.float64]:
    """
    Extract wave points from OHLC financial data.

    This function processes OHLC data to extract price points based on candlestick patterns,
    and removes consecutive points that follow the same trend direction.

    Parameters
    ----------
    open : numpy.ndarray
        Array of opening prices
    high : numpy.ndarray
        Array of high prices
    low : numpy.ndarray
        Array of low prices
    close : numpy.ndarray
        Array of closing prices

    Returns
    -------
    numpy.ndarray
        2D array of wave points with shape (n, 2), where each row contains [index, price]

    Notes
    -----
    The algorithm works as follows:
    1. For each candle:
       - If close > open: adds low then high to the price list
       - If close < open: adds high then low to the price list
    2. Removes intermediate points where three consecutive points form a consistent trend
       (either all increasing or all decreasing)

    Raises
    ------
    ValueError
        If the price arrays have different lengths
    """
    if not (len(open) == len(high) == len(low) == len(close)):
        raise ValueError("All price arrays must have the same length")
        
    open_prices = np.asarray(open, dtype=np.float64)
    high_prices = np.asarray(high, dtype=np.float64)
    low_prices = np.asarray(low, dtype=np.float64)
    close_prices = np.asarray(close, dtype=np.float64)
    
    indices_list: List[int] = []
    prices_list: List[float] = []
    
    for i in range(len(close_prices)):
        if close_prices[i] >= open_prices[i]:
            indices_list.extend([i, i])
            prices_list.extend([low_prices[i], high_prices[i]])
        else:
            indices_list.extend([i, i])
            prices_list.extend([high_prices[i], low_prices[i]])
    
    indices_array = np.array(indices_list)
    prices_array = np.array(prices_list)
    
    if len(prices_array) >= 3:
        keep_mask = np.ones(len(prices_array), dtype=bool)
        
        for i in range(1, len(prices_array) - 1):
            if ((prices_array[i-1] < prices_array[i] < prices_array[i+1]) or 
                (prices_array[i-1] > prices_array[i] > prices_array[i+1])):
                keep_mask[i] = False
        
        indices_array = indices_array[keep_mask]
        prices_array = prices_array[keep_mask]
    
    return prices_array

def zigzag(
    prices: ArrayLike, 
    threshold: float = 0.03
) -> NDArray[np.float64]:
    """
    Extract zigzag pivot points from price data based on a percentage threshold.
    
    Parameters
    ----------
    prices : ArrayLike
        1D array/list of price values or 2D array/list of [index, price] points
    threshold : float, default 0.03
        Minimum percentage change required to identify a new pivot point (0.03 = 3%)
        
    Returns
    -------
    NDArray[np.float64]
        2D array of zigzag points with shape (n, 2), where each row contains [index, price]
        
    Notes
    -----
    The algorithm identifies significant price movements while filtering out
    minor fluctuations. It marks pivot points where the price changes direction
    by at least the specified threshold percentage.

    Raises
    ------
    ValueError
        If the price arrays have different lengths
    """
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    if prices.ndim == 1:
        indices_array = np.arange(len(prices))
        price_values = prices.astype(np.float64)
    else:
        indices_array = prices[:, 0].astype(np.int64)
        price_values = prices[:, 1].astype(np.float64)
    
    pivot_indices = []
    pivot_prices = []
    
    if len(price_values) == 1:
        return prices
    
    if len(price_values) == 0:
        return np.zeros((0, 2))
    
    pivot_indices.append(indices_array[0])
    pivot_prices.append(price_values[0])
    
    last_direction = 0
    
    extreme_price = price_values[0]
    extreme_index = indices_array[0]
    
    for i in range(1, len(price_values)):
        current_price = price_values[i]
        current_index = indices_array[i]
        
        percent_change = (current_price - extreme_price) / extreme_price
        
        current_direction = 1 if current_price > extreme_price else -1
        
        if last_direction == 0:
            if abs(percent_change) >= threshold:
                last_direction = current_direction
                pivot_indices[-1] = extreme_index
                pivot_prices[-1] = extreme_price
                pivot_indices.append(current_index)
                pivot_prices.append(current_price)
                extreme_price = current_price
                extreme_index = current_index
            else:
                if current_direction == 1: 
                    extreme_price = current_price
                    extreme_index = current_index
                elif current_direction == -1 and current_price < extreme_price: 
                    extreme_price = current_price
                    extreme_index = current_index
        else:
            if abs(percent_change) >= threshold and current_direction != last_direction:
                pivot_indices.append(extreme_index)
                pivot_prices.append(extreme_price)
                last_direction = current_direction
                extreme_price = current_price
                extreme_index = current_index
            elif current_direction == last_direction and (
                (last_direction == 1 and current_price > extreme_price) or
                (last_direction == -1 and current_price < extreme_price)
            ):
                extreme_price = current_price
                extreme_index = current_index
    
    if extreme_index != pivot_indices[-1]:
        pivot_indices.append(extreme_index)
        pivot_prices.append(extreme_price)
    
    result = np.column_stack((pivot_indices, pivot_prices))
    
    return result
