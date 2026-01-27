from __future__ import annotations

from array_api._2024_12 import Array
from array_api_compat import array_namespace


def periodic_difference(t: Array, tau: Array) -> Array:
    xp = array_namespace(t, tau)
    two_pi = 2 * xp.pi
    return xp.remainder(tau - t + xp.pi, two_pi) - xp.pi


def is_close(t: Array, tau: Array, eps: float) -> Array:
    xp = array_namespace(t, tau)
    delta = periodic_difference(t, tau)
    return xp.abs(delta) <= eps
