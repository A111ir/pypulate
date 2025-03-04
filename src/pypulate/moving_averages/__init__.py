"""
Moving Averages Module

This module provides various moving average implementations for financial time series analysis.
All functions use numpy arrays for input and output to ensure high performance.

This module offers two ways to use moving averages:
1. Function calls: sma(data, period=9)
2. Method chaining: as_ts(data).sma(9).ema(14)
"""

from .movingaverages import (
    # Simple moving averages
    sma, ema, wma, tma, smma, zlma, hma,
    
    # Specialized moving averages
    vwma, kama, alma, frama, jma, lsma, mcginley_dynamic, 
    t3, vama, laguerre_filter, modular_filter, rdma,

)

__all__ = [
    # Simple moving averages
    'sma', 'ema', 'wma', 'tma', 'smma', 'zlma', 'hma',
    
    # Specialized moving averages
    'vwma', 'kama', 'alma', 'frama', 'jma', 'lsma', 'mcginley_dynamic',
    't3', 'vama', 'laguerre_filter', 'modular_filter', 'rdma',
]
