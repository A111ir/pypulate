"""
Pypulate: A Python package for financial time series analysis and business KPIs

This package provides tools for analyzing financial time series data,
including wave pattern detection, zigzag identification, various
moving average calculations, and business KPIs.
"""

# Import submodules
from . import moving_averages
from . import transforms
from . import dtypes
from . import kpi
from . import filters
from . import technical

# Import Parray for easy access
from .dtypes.parray import Parray
from .dtypes.portfolio import Portfolio
from .dtypes.kpi import KPI
from .dtypes.service_pricing import ServicePricing

# Define package metadata
__version__ = "0.1.0"
__author__ = "Amir Rezaei"
__email__ = "corvology@gmail.com"

__all__ = [
    'moving_averages',
    'transforms',
    'dtypes',
    'kpi',
    'filters',
    'technical',
    'wave',
    'zigzag',
    'Parray',
    'Portfolio',
    'KPI',
    'ServicePricing'
]
