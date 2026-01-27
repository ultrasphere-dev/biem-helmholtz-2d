from __future__ import annotations

import math
from typing import Any

import pytest

from biem_helmholtz_2d._potential import D_t, dlp, slp
from biem_helmholtz_2d._quadrature import log_cot_power_quadrature, trapezoidal_quadrature
from biem_helmholtz_2d._shape import CircleShape
from biem_helmholtz_2d._scipy_wrapper import scipy_hankel1, scipy_jv

@pytest.mark.parametrize("t", [00, 1, 2])
@pytest.mark.parametrize("rho", [0.7, 1.3])
def test_D_t_diagonal_limit_circle(xp: Any, device: Any, dtype: Any, rho: float, t: float) -> None:
    shape = CircleShape(rho)
    actual = D_t(
        xp.asarray(t, device=device, dtype=dtype),
        xp.asarray(t, device=device, dtype=dtype),
        shape.x,
        shape.dx,
        shape.ddx,
        eps=xp.inf
    )
    actual_numerical = D_t(
       xp.asarray(t, device=device, dtype=dtype),
         xp.asarray(t + 1e-6, device=device, dtype=dtype), 
        shape.x,
        shape.dx,
        shape.ddx,
        eps=0
     )
    assert actual == pytest.approx(actual_numerical, rel=1e-3)
    assert actual == pytest.approx(-0.5, rel=1e-6)


@pytest.mark.parametrize("n", [8, 10, 128])
@pytest.mark.parametrize("tau", [0, 0.11])
@pytest.mark.parametrize("kernel_kind", ["slp", "dlp"])
@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("rho", [1.0])
@pytest.mark.parametrize("k", [1.0])
def test_circle_case_matches_theorem(
    xp: Any, device: Any, dtype: Any, kernel_kind: str, m: int, rho: float, k: float, n: int, tau: float
) -> None:
    from scipy.special import hankel1, jv, jvp

    t, w = trapezoidal_quadrature(n, t_start=tau, xp=xp, device=device, dtype=dtype)
    _, r = log_cot_power_quadrature(n, 0, t_start=tau, xp=xp, device=device, dtype=dtype)
    shape = CircleShape(rho)

    # actual
    t += tau # "fix" quadrature by ∫w(t-tau)f(t) -> ∫w(t)f(t+tau)
    exp_mt = xp.exp(1j * m * t)
    exp_mt_tau = xp.exp(1j * m * xp.asarray(tau, device=device, dtype=dtype))

    if kernel_kind == "slp":
        log_coeff, analytic = slp(t, tau, k, shape.x, shape.dx, eps=0.0)
    else:
        log_coeff, analytic = dlp(t, tau, k, shape.x, shape.dx, shape.ddx, eps=0.0)

    actual = xp.sum(exp_mt * (r * log_coeff + w * analytic))

    # expected
    kr = k * rho
    abs_m = abs(m)
    kr_array = xp.asarray(kr, device=device, dtype=dtype)
    h_abs = scipy_hankel1(abs_m, kr_array)
    j_abs = scipy_jv(abs_m, kr_array, 0)
    jprime_abs = scipy_jv(abs_m, kr_array, 1)

    if kernel_kind == "slp":
        expected = (1j * math.pi * rho / 2) * h_abs * j_abs * exp_mt_tau
    else:
        expected = (1j * math.pi * k * rho / 2) * h_abs * jprime_abs * exp_mt_tau
        expected -= 0.5 * exp_mt_tau

    actual, expected = complex(actual), complex(expected)
    assert actual == pytest.approx(expected, rel=5e-6)
