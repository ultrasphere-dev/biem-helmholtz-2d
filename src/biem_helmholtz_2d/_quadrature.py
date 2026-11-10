"""
Bounded caches for the fixed quadrature rules used by this package.

The returned arrays are shared and must not be modified in place.
"""

from functools import lru_cache

from ie_circle import log_cot_power_quadrature as _log_cot_power_quadrature
from ie_circle import trapezoidal_quadrature as _trapezoidal_quadrature

trapezoidal_quadrature = lru_cache(maxsize=32)(_trapezoidal_quadrature)
log_cot_power_quadrature = lru_cache(maxsize=32)(_log_cot_power_quadrature)
