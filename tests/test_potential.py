from __future__ import annotations

import math
from typing import Any

import pytest

from biem_helmholtz_2d._potential import D_t, dlp, slp
from biem_helmholtz_2d._quadrature import log_cot_power_quadrature, trapezoidal_quadrature
from biem_helmholtz_2d._shape import CircleShape


@pytest.mark.parametrize("rho", [0.7, 1.3])
def test_D_t_diagonal_limit_circle(xp: Any, device: Any, dtype: Any, rho: float) -> None:
    n = 16
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    shape = CircleShape(rho)
    x, dx, ddx = shape.x, shape.dx, shape.ddx
    tau = t

    vals = D_t(t, tau, x, dx, ddx, eps=1e-8)
    target = xp.asarray(0.5, device=device, dtype=vals.dtype)
    err = xp.max(xp.abs(vals - target))
    assert err < 1e-12


@pytest.mark.parametrize("tau_val", [0.2, 1.1])
def test_D_t_matches_geometric_definition(
    xp: Any, device: Any, dtype: Any, tau_val: float
) -> None:
    n = 12
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    shape = CircleShape(1.1)
    x, dx, ddx = shape.x, shape.dx, shape.ddx
    tau = xp.asarray(tau_val, device=device, dtype=dtype)

    vals = D_t(t, tau, x, dx, ddx, eps=0.0)

    x_t = x(t)
    x_tau = x(tau)
    diff = x_tau - x_t
    normal = xp.stack([-dx(t)[..., 1], dx(t)[..., 0]], axis=-1)
    expected = xp.sum(normal * diff, axis=-1) / xp.sum(diff**2, axis=-1)

    err = xp.max(xp.abs(vals - expected))
    assert err < 1e-12


@pytest.mark.parametrize("kernel_kind", ["slp", "dlp"])
@pytest.mark.parametrize("rho", [0.6, 1.0])
def test_split_reconstructs_kernel(
    xp: Any, device: Any, dtype: Any, kernel_kind: str, rho: float
) -> None:
    from scipy.special import hankel1

    n = 24
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    shape = CircleShape(rho)
    x, dx, ddx = shape.x, shape.dx, shape.ddx

    if kernel_kind == "slp":
        tau = 0.4
        k = 2.0
        log_coeff, analytic = slp(t, tau, k, x, dx, eps=0.0)
        diff = x(xp.asarray(tau, device=device, dtype=dtype)) - x(t)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        h_ref = xp.asarray(hankel1(0, xp.asarray(k * r, device="cpu")), device=device)
        jac_t = xp.sqrt(xp.sum(dx(t) ** 2, axis=-1))
        kernel_ref = (1j / 4) * h_ref * jac_t
    else:
        tau = 0.9
        k = 2.0
        log_coeff, analytic = dlp(t, tau, k, x, dx, ddx, eps=0.0)
        x_tau = x(xp.asarray(tau, device=device, dtype=dtype))
        diff = x_tau - x(t)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        hankel_val = xp.asarray(hankel1(1, xp.asarray(k * r, device="cpu")), device=device)
        d_t = D_t(t, xp.asarray(tau, device=device, dtype=dtype), x, dx, ddx, eps=0.0)
        kernel_ref = (1j / 4) * (k * r) * hankel_val * d_t

    log_kernel = xp.log(4 * xp.sin((t - tau) / 2) ** 2)
    split_kernel = log_coeff * log_kernel + analytic

    denom = xp.max(xp.asarray([xp.max(xp.abs(kernel_ref)), 1], device=device))
    rel_err = xp.max(xp.abs(split_kernel - kernel_ref)) / denom
    assert rel_err < 5e-8


@pytest.mark.parametrize("kernel_kind", ["slp", "dlp"])
@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("rho", [0.6, 1.0])
def test_circle_case_matches_theorem(
    xp: Any, device: Any, dtype: Any, kernel_kind: str, m: int, rho: float
) -> None:
    from scipy.special import hankel1, jv, jvp

    tau = 0.37
    n = 256
    t, w = trapezoidal_quadrature(n, t_start=tau, xp=xp, device=device, dtype=dtype)
    _, r = log_cot_power_quadrature(n, 0, t_start=tau, xp=xp, device=device, dtype=dtype)
    shape = CircleShape(rho)
    x, dx, ddx = shape.x, shape.dx, shape.ddx

    k = 2.0
    abs_m = abs(m)
    exp_mt = xp.exp(1j * m * t)
    exp_mt_tau = xp.exp(1j * m * xp.asarray(tau, device=device, dtype=dtype))

    if kernel_kind == "slp":
        log_coeff, analytic = slp(t, tau, k, x, dx, eps=0.0)
    else:
        log_coeff, analytic = dlp(t, tau, k, x, dx, ddx, eps=0.0)

    integral = xp.sum(exp_mt * (r * log_coeff + w * analytic))

    kr = k * rho
    h_abs = xp.asarray(hankel1(abs_m, kr), device=device, dtype=dtype)
    j_abs = xp.asarray(jv(abs_m, kr), device=device, dtype=dtype)
    jprime_abs = xp.asarray(jvp(abs_m, kr, 1), device=device, dtype=dtype)

    if kernel_kind == "slp":
        expected = (1j * math.pi * rho / 2) * h_abs * j_abs * exp_mt_tau
    else:
        expected = (1j * math.pi * k * rho / 2) * h_abs * jprime_abs * exp_mt_tau
        integral = 0.5 * exp_mt_tau + integral

    denom = xp.max(xp.asarray([xp.abs(expected), 1], device=device, dtype=dtype))
    rel_err = xp.abs(integral - expected) / denom
    assert rel_err < 5e-6
