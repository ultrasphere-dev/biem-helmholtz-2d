from __future__ import annotations

import math
from typing import Any
from array_api_compat import array_namespace
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull

from biem_helmholtz_2d._hankel import neumann_y1_y2
from biem_helmholtz_2d._quadrature import (
    log_cot_power_quadrature,
    trapezoidal_quadrature,
)


def _scipy_yv(
    order: int,
    x: Array,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> Array:
    from scipy.special import yv

    return xp.asarray(yv(order, xp.asarray(x, device="cpu")), device=device, dtype=dtype)


def _f(x: Array) -> Array:
    xp = array_namespace(x)
    return 4 * xp.sin(x / 2)


def _g(x: Array) -> Array:
    xp = array_namespace(x)
    return 3 + (xp.cos(x) / 4) + xp.cos(4 * x)


@pytest.mark.parametrize("t_start_factor", [0, 0.5])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_neumann_split_quadrature_matches_trapezoidal(
    xp: Any, device: Any, dtype: Any, order: int, t_start_factor: float
) -> None:
    """
    Test if $\int_0^{2\pi} g(t) f(t)^{order} Y_{order}(f(t)) dt$ computed by
    the Neumann split quadrature matches the reference trapezoidal rule.
    """
    # approx
    n = 24
    x, w = trapezoidal_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    x, r = log_cot_power_quadrature(
        n, 0, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )

    fprime0 = xp.asarray(2, device=device, dtype=dtype) if order == 0 else None
    t_s = xp.asarray(0, device=device, dtype=dtype)
    y1, y2 = neumann_y1_y2(
        x,
        order=order,
        f=_f,
        fprime0=fprime0,
        eps=0,
        t_singularity=t_s,
    )

    g_vals = _g(x)
    approx = xp.sum(g_vals * (r * y1 + w * y2))

    # expected
    n_ref = 150000 if order == 0 else 1000
    t_ref, w_ref = trapezoidal_quadrature(
        n_ref,
        t_start_factor=0.5,
        xp=xp,
        device=device,
        dtype=dtype,
    )
    f_ref = _f(t_ref)
    y_ref = _scipy_yv(order, f_ref, xp=xp, device=device, dtype=dtype)
    integrand = _g(t_ref) * (f_ref**order) * y_ref
    reference = xp.sum(w_ref * integrand)

    # compare
    assert approx == pytest.approx(reference, rel=1e-4 if order == 0 else 1e-6)
