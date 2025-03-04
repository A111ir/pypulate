"""
Wave and zigzag transforms for financial time series data.

This module provides functions for extracting wave points and zigzag patterns
from financial time series data, which are useful for technical analysis.
"""

import numpy as np
from typing import Tuple, Optional, Union, List

def wave(
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
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
    """
    # Ensure all arrays have the same length
    if not (len(open) == len(high) == len(low) == len(close)):
        raise ValueError("All price arrays must have the same length")
        
    # Convert to float64 to ensure consistent calculations
    open_prices = np.asarray(open, dtype=np.float64)
    high_prices = np.asarray(high, dtype=np.float64)
    low_prices = np.asarray(low, dtype=np.float64)
    close_prices = np.asarray(close, dtype=np.float64)
    
    # Initialize lists to store wave points
    indices_list: List[int] = []
    prices_list: List[float] = []
    
    # Process each candle to extract initial wave points
    for i in range(len(close_prices)):
        if close_prices[i] >= open_prices[i]:
            # Bullish or neutral candle: add low then high
            indices_list.extend([i, i])
            prices_list.extend([low_prices[i], high_prices[i]])
        else:
            # Bearish candle: add high then low
            indices_list.extend([i, i])
            prices_list.extend([high_prices[i], low_prices[i]])
    
    # Convert to numpy arrays
    indices_array = np.array(indices_list)
    prices_array = np.array(prices_list)
    
    # Remove intermediate points in consistent trends
    if len(prices_array) >= 3:
        keep_mask = np.ones(len(prices_array), dtype=bool)
        
        for i in range(1, len(prices_array) - 1):
            # Check if three consecutive points form a consistent trend
            if ((prices_array[i-1] < prices_array[i] < prices_array[i+1]) or 
                (prices_array[i-1] > prices_array[i] > prices_array[i+1])):
                keep_mask[i] = False
        
        # Apply the mask to keep only the relevant points
        indices_array = indices_array[keep_mask]
        prices_array = prices_array[keep_mask]
    
    return prices_array

def zigzag(
    prices: Union[np.ndarray, List[float], List[List[float]]], 
    threshold: float = 0.03
) -> np.ndarray:
    """
    Extract zigzag pivot points from price data based on a percentage threshold.
    
    Parameters
    ----------
    prices : numpy.ndarray or list
        1D array/list of price values or 2D array/list of [index, price] points
    threshold : float, default 0.03
        Minimum percentage change required to identify a new pivot point (0.03 = 3%)
        
    Returns
    -------
    numpy.ndarray
        2D array of zigzag points with shape (n, 2), where each row contains [index, price]
        
    Notes
    -----
    The algorithm identifies significant price movements while filtering out
    minor fluctuations. It marks pivot points where the price changes direction
    by at least the specified threshold percentage.
    """
    # Convert list to numpy array if needed
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices)
    
    # Handle 1D and 2D arrays
    if prices.ndim == 1:
        # If 1D array, create indices
        indices_array = np.arange(len(prices))
        price_values = prices.astype(np.float64)
    else:
        # If 2D array, extract indices and prices
        indices_array = prices[:, 0].astype(np.int64)
        price_values = prices[:, 1].astype(np.float64)
    
    # Initialize arrays to track pivot points
    pivot_indices = []
    pivot_prices = []
    
    # Need at least 2 points to find pivots
    if len(price_values) == 1:
        # Return empty 2D array with shape (0, 2)
        return prices
    
    if len(price_values) == 0:
        return np.zeros((0, 2))
    
    # Initialize with the first point
    pivot_indices.append(indices_array[0])
    pivot_prices.append(price_values[0])
    
    # Initialize direction: 1 for up, -1 for down, 0 for undetermined
    last_direction = 0
    
    # Track the highest/lowest price seen so far
    extreme_price = price_values[0]
    extreme_index = indices_array[0]
    
    # Process each price point
    for i in range(1, len(price_values)):
        current_price = price_values[i]
        current_index = indices_array[i]
        
        # Calculate percentage change from the extreme price
        percent_change = (current_price - extreme_price) / extreme_price
        
        # Determine direction based on the current price compared to extreme
        current_direction = 1 if current_price > extreme_price else -1
        
        # If direction is undetermined, set it based on the first significant move
        if last_direction == 0:
            if abs(percent_change) >= threshold:
                last_direction = current_direction
                # Add the extreme point as the first real pivot
                pivot_indices[-1] = extreme_index
                pivot_prices[-1] = extreme_price
                # Add the current point as the next pivot
                pivot_indices.append(current_index)
                pivot_prices.append(current_price)
                # Update extreme to current point
                extreme_price = current_price
                extreme_index = current_index
            else:
                # Update extreme if needed but don't change direction yet
                if current_direction == 1:  # New high
                    extreme_price = current_price
                    extreme_index = current_index
                elif current_direction == -1 and current_price < extreme_price:  # New low
                    extreme_price = current_price
                    extreme_index = current_index
        else:
            # If we have a direction and the current move is significant and opposite
            if abs(percent_change) >= threshold and current_direction != last_direction:
                # Add the extreme point as a pivot
                pivot_indices.append(extreme_index)
                pivot_prices.append(extreme_price)
                # Update direction
                last_direction = current_direction
                # Update extreme to current point
                extreme_price = current_price
                extreme_index = current_index
            # If the current point extends the current trend, update the extreme
            elif current_direction == last_direction and (
                (last_direction == 1 and current_price > extreme_price) or
                (last_direction == -1 and current_price < extreme_price)
            ):
                extreme_price = current_price
                extreme_index = current_index
    
    # Add the last extreme point if it's not already added
    if extreme_index != pivot_indices[-1]:
        pivot_indices.append(extreme_index)
        pivot_prices.append(extreme_price)
    
    # Combine indices and prices into a 2D array
    result = np.column_stack((pivot_indices, pivot_prices))
    
    return result
