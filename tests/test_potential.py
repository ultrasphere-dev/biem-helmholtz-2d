from __future__ import annotations

import math
from typing import Any, Literal

import array_api_extra as xpx
import numpy as np
import pytest
from array_api.latest import Array
from ie_circle import (
    CircleShape,
    KressShape,
    log_cot_power_quadrature,
    nystrom,
    shift_quadrature_singularity,
    trapezoidal_quadrature,
)
from ie_circle._bie import QuadratureType

from biem_helmholtz_2d._potential import A1, dlp_kernel_split, slp_kernel_split
from biem_helmholtz_2d._scipy_wrapper import scipy_hankel1, scipy_jv


@pytest.mark.parametrize("type", ["slp", "dlp"])
@pytest.mark.parametrize("singularity", ["log", "cont"])
def test_potential_match_known_values_kress_shape(
    xp: Any,
    device: Any,
    dtype: Any,
    type: Literal["slp", "dlp"],
    singularity: Literal["log", "cont"],
) -> None:
    shape = KressShape()
    t = trapezoidal_quadrature(3, xp=xp, device=device, dtype=dtype)[0]
    k = xp.asarray(1.0, device=device, dtype=dtype)
    t, tau = t[:, None], t[None, :]
    if type == "slp":
        actual = slp_kernel_split(t, tau, k, shape.x, shape.dx, eps=0.0)[
            0 if singularity == "log" else 1
        ]
    else:
        actual = dlp_kernel_split(t, tau, k, shape.x, shape.dx, shape.ddx, eps=0.0)[
            0 if singularity == "log" else 1
        ]
    expected = xp.asarray(
        np.loadtxt(
            f"tests/kress_shape/{type}_{singularity}.csv", delimiter=",", dtype=np.complex128
        ),
        device=device,
        dtype=xp.result_type(dtype, 1j),
    )
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-6))


@pytest.mark.parametrize("t", [00, 1, 2])
@pytest.mark.parametrize("rho", [0.7, 1.3])
def test_D_t_diagonal_limit_circle(xp: Any, device: Any, dtype: Any, rho: float, t: float) -> None:
    shape = CircleShape(rho)
    actual = A1(
        xp.asarray(t, device=device, dtype=dtype),
        xp.asarray(t, device=device, dtype=dtype),
        shape.x,
        shape.dx,
        shape.ddx,
        eps=xp.inf,
    )
    actual_numerical = A1(
        xp.asarray(t, device=device, dtype=dtype),
        xp.asarray(t + 1e-6, device=device, dtype=dtype),
        shape.x,
        shape.dx,
        shape.ddx,
        eps=0,
    )
    assert actual == pytest.approx(actual_numerical, rel=1e-3)
    assert actual == pytest.approx(-0.5, rel=1e-6)


@pytest.mark.parametrize("t_start_factor", [0, 0.5])
@pytest.mark.parametrize("n", [8, 10, 128])
@pytest.mark.parametrize("tau", [0, 0.11])
@pytest.mark.parametrize("kernel_kind", ["slp", "dlp"])
@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("rho", [1.0])
@pytest.mark.parametrize("k", [1.0])
def test_circle_case_matches_theorem(
    xp: Any,
    device: Any,
    dtype: Any,
    kernel_kind: str,
    m: int,
    rho: float,
    k: float,
    n: int,
    tau: float,
    t_start_factor: float,
) -> None:
    t, w = trapezoidal_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    t, r = shift_quadrature_singularity(log_cot_power_quadrature, tau)(
        n, 0, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    tau = xp.asarray(tau, device=device, dtype=dtype)
    shape = CircleShape(rho)

    # actual
    exp_mt = xp.exp(1j * m * t)
    exp_mt_tau = xp.exp(1j * m * xp.asarray(tau, device=device, dtype=dtype))

    if kernel_kind == "slp":
        log_coeff, analytic = slp_kernel_split(t, tau, k, shape.x, shape.dx, eps=0.0)
    else:
        log_coeff, analytic = dlp_kernel_split(t, tau, k, shape.x, shape.dx, shape.ddx, eps=0.0)

    actual = xp.sum(exp_mt * (r * log_coeff + w * analytic))

    # expected
    abs_m = abs(m)
    kr = xp.asarray(k * rho, device=device, dtype=dtype)
    h_abs_kr = scipy_hankel1(abs_m, kr)
    j_abs_kr = scipy_jv(abs_m, kr, 0)
    jprime_abs_kr = scipy_jv(abs_m, kr, 1)

    if kernel_kind == "slp":
        expected = (1j * math.pi * rho / 2) * h_abs_kr * j_abs_kr * exp_mt_tau
    else:
        expected = (1j * math.pi * k * rho / 2) * h_abs_kr * jprime_abs_kr * exp_mt_tau
        expected -= 0.5 * exp_mt_tau

    actual, expected = complex(actual), complex(expected)
    assert actual == pytest.approx(expected, rel=5e-6)


@pytest.mark.parametrize("n", [8, 10, 128])
@pytest.mark.parametrize("kernel_kind", ["slp", "dlp"])
@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("rho", [1.0])
@pytest.mark.parametrize("k", [1.0])
def test_circle_sol_matches_theorem(
    xp: Any,
    device: Any,
    dtype: Any,
    kernel_kind: str,
    m: int,
    rho: float,
    k: float,
    n: int,
) -> None:
    shape = CircleShape(rho)
    abs_m = abs(m)
    k = xp.asarray(k, device=device, dtype=dtype)
    kr = k * rho
    h_abs_kr = scipy_hankel1(abs_m, kr)
    j_abs_kr = scipy_jv(abs_m, kr, 0)
    jprime_abs_kr = scipy_jv(abs_m, kr, 1)
    if kernel_kind == "slp":

        def a(t: Array) -> Array:
            return xp.zeros_like(t)[..., None]

        def rhs(t: Array) -> Array:
            return xp.asarray((1j * math.pi * rho) / 2 * h_abs_kr * j_abs_kr * xp.exp(1j * m * t))[
                ..., None
            ]

        def k_log(t: Array, tau: Array) -> Array:
            log_coeff, _ = slp_kernel_split(t, tau, k, shape.x, shape.dx, eps=0.0)
            return log_coeff[..., None, None]

        def k_cont(t: Array, tau: Array) -> Array:
            _, analytic = slp_kernel_split(t, tau, k, shape.x, shape.dx, eps=0.0)
            return analytic[..., None, None]
    else:

        def a(t: Array) -> Array:
            return xp.ones_like(t)[..., None] / 2

        def rhs(t: Array) -> Array:
            return xp.asarray(
                (1j * math.pi * k * rho) / 2 * h_abs_kr * jprime_abs_kr * xp.exp(1j * m * t)
            )[..., None]

        def k_log(t: Array, tau: Array) -> Array:
            log_coeff, _ = dlp_kernel_split(t, tau, k, shape.x, shape.dx, shape.ddx, eps=0.0)
            return log_coeff[..., None, None]

        def k_cont(t: Array, tau: Array) -> Array:
            _, analytic = dlp_kernel_split(t, tau, k, shape.x, shape.dx, shape.ddx, eps=0.0)
            return analytic[..., None, None]

    kernels = {
        (QuadratureType.NO_SINGULARITY, 0): k_cont,
        (QuadratureType.LOG_COT_POWER, 0): k_log,
    }
    density = nystrom(a, kernels, rhs, n=n, xp=xp, device=device, dtype=dtype)
    eval_points = xp.random.random_uniform(shape=(3,), device=device, dtype=dtype) * 2 * math.pi
    actual = density(eval_points)
    expected = xp.exp(1j * m * eval_points)
    assert xp.all(xpx.isclose(actual, expected, rtol=5e-6))
