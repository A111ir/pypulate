"""
Pypulate: A Python package for financial modeling and simulation.

This package provides tools for portfolio analysis, risk assessment,
asset allocation, and credit scoring.
"""

# Import submodules
from . import moving_averages
from . import transforms
from . import dtypes
from . import kpi
from . import filters
from . import technical
from . import credit
from . import asset
from . import preprocessing

# Import Parray for easy access
from .dtypes.parray import Parray
from .dtypes.portfolio import Portfolio
from .dtypes.kpi import KPI
from .dtypes.service_pricing import ServicePricing
from .dtypes.allocation import Allocation
from .dtypes.credit_scoring import CreditScoring

# Define package metadata
__version__ = "0.3.0"
__author__ = "Amir Rezaei"
__email__ = "corvology@gmail.com"

__all__ = [
    'moving_averages',
    'transforms',
    'dtypes',
    'kpi',
    'filters',
    'technical',
    'credit',
    'asset',
    'processing',
    'Parray',
    'Portfolio',
    'KPI',
    'ServicePricing',
    'Allocation',
    'CreditScoring',
    'preprocessing',
    'statistics',
]
