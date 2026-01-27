from __future__ import annotations

import math
from typing import Any

import pytest
from scipy import integrate, special
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from biem_helmholtz_2d._hankel import hankel_h1_h2, neumann_y1_y2
from biem_helmholtz_2d._quadrature import (
    log_cot_power_quadrature,
    trapezoidal_quadrature,
)
from biem_helmholtz_2d._scipy_wrapper import scipy_hankel1, scipy_yv


def _g(x: Array) -> Array:
    xp = array_namespace(x)
    return 43 + (xp.sin(x - 0.3) / 4) + xp.cos(4 * x)

@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("t_start_factor", [0, 0.55555])
@pytest.mark.parametrize("split_type", ["neumann", "hankel"])
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("t_singularity", [0, 0.11383759])
def test_split_quadrature_matches_trapezoidal(
    xp: Any, device: Any, dtype: Any, n: int, order: int, split_type: str, t_start_factor: float, t_singularity: float
) -> None:
    r"""
    Test if $\int_0^{2\pi} g(t) f(t)^{order} Y_{order}(f(t)) dt$ or
    $\int_0^{2\pi} g(t) f(t)^{order} H_{order}^{(1)}(f(t)) dt$ computed by
    the split quadrature matches the reference trapezoidal rule when the
    singularity is placed at $t_s \in \{0, 0.11\}$.
    """

    def _f(x_in: Array) -> Array:
        xp_local = array_namespace(x_in)
        return 4 * xp_local.sin(xp_local.abs((x_in - t_singularity)) / 2)

    # approx
    x, w = trapezoidal_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    x, r = log_cot_power_quadrature(
        n, 0, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )

    fprime0 = xp.asarray(2, device=device, dtype=dtype) if order == 0 else None
    y1, y2 = (hankel_h1_h2 if split_type == "hankel" else neumann_y1_y2)(
            x + t_singularity,
            order=order,
            f=_f,
            fprime0=fprime0,
            eps=0.0001,
            t_singularity=t_singularity,
        )
    g_vals = _g(x + t_singularity)
    actual = xp.sum(g_vals * (r * y1 + w * y2))
    actual = complex(actual)

    # expected
    def integrand_func(t: float, real: bool) -> float:
        import numpy as np

        t_ref = np.asarray(t)
        f_ref = _f(t_ref)
        y_ref = (scipy_yv if split_type == "neumann" else scipy_hankel1)(order, f_ref)
        res = (_g(t_ref) * (f_ref**order) * y_ref).item()
        return res.real if real else res.imag
        
    expected_real, _ = integrate.quad(lambda t: integrand_func(t, True), 1e-8, 2 * math.pi - 1e-8, limit=2000)
    expected_imag, _ = integrate.quad(lambda t: integrand_func(t, False), 1e-8, 2 * math.pi - 1e-8, limit=2000)

    # compare
    assert actual.real == pytest.approx(expected_real, rel=1e-6)
    assert actual.imag == pytest.approx(expected_imag, rel=1e-6)