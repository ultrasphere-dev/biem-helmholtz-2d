from __future__ import annotations

import math
from typing import Any

import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull

from biem_helmholtz_2d.hankel import neumann_y1_y2
from biem_helmholtz_2d.quadrature import (
    log_cot_power_shifted_quadrature,
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


def _f(x, xp):
    return 2 * xp.sin(x / 2)


def _g(x, xp):
    return 1 + (xp.cos(x) / 4)


@pytest.mark.parametrize("t_start_factor", [0, 0.5])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_neumann_split_quadrature_matches_trapezoidal(
    xp: Any, device: Any, dtype: Any, order: int, t_start_factor: float
) -> None:
    n = 24
    x, w = trapezoidal_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    _, r = log_cot_power_shifted_quadrature(
        n, 0, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )

    fprime0 = xp.ones_like(x[0])
    if order == 0:
        y1, y2 = neumann_y1_y2(
            x,
            order,
            lambda t: _f(t, xp),
            fprime0,
            0,
        )
    else:
        y1, y2 = neumann_y1_y2(
            x,
            order,
            lambda t: _f(t, xp),
            None,
            0,
        )

    g_vals = _g(x, xp)
    approx = xp.sum(g_vals * (r * y1 + w * y2))

    n_ref = 1024
    h_ref = (2 * math.pi) / (2 * n_ref - 1)
    t_ref, w_ref = trapezoidal_quadrature(
        n_ref,
        t_start=h_ref / 2,
        xp=xp,
        device=device,
        dtype=dtype,
    )
    f_ref = _f(t_ref, xp)
    y_ref = _scipy_yv(order, f_ref, xp=xp, device=device, dtype=dtype)
    integrand = _g(t_ref, xp) * (f_ref**order) * y_ref
    reference = xp.sum(w_ref * integrand)

    rel_err = xp.abs(approx - reference) / xp.max(
        xp.asarray([xp.abs(reference), 1], device=device, dtype=dtype)
    )
    if order == 0:
        tol = 5e-3
    else:
        tol = 5e-6
    assert rel_err < tol
