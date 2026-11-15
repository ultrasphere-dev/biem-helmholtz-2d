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


@pytest.mark.parametrize("kind", ["slp", "dlp"])
@pytest.mark.parametrize("part", ["log", "cont"])
def test_potential_match_known_values_kress_shape(
    xp: Any, device: Any, dtype: Any, kind: str, part: str
) -> None:
    shape = KressShape()
    t = trapezoidal_quadrature(3, xp=xp, device=device, dtype=dtype)[0]
    k = xp.asarray(1, device=device, dtype=dtype)
    t, tau = t[:, None], t[None, :]
    values = {
        "t": t,
        "tau": tau,
        "k": k,
        "x_t": shape.x(t)[..., 0, :],
        "x_tau": shape.x(tau)[..., 0, :],
        "dx_tau": shape.dx(tau)[..., 0, :],
    }
    if kind == "slp":
        actual = slp_kernel_split(**values)[0 if part == "log" else 1]
    else:
        actual = dlp_kernel_split(**values, ddx_tau=shape.ddx(tau)[..., 0, :])[
            0 if part == "log" else 1
        ]
    expected = xp.asarray(
        np.loadtxt(f"tests/kress_shape/{kind}_{part}.csv", delimiter=",", dtype=np.complex128),
        device=device,
        dtype=xp.result_type(dtype, 1j),
    )
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-6))


@pytest.mark.parametrize("t_value", [0, 1, 2])
@pytest.mark.parametrize("rho", [0.7, 1.3])
def test_D_t_diagonal_limit_circle(
    xp: Any, device: Any, dtype: Any, rho: float, t_value: float
) -> None:
    shape = CircleShape(rho)
    t = xp.asarray(t_value, device=device, dtype=dtype)
    tau = t + 1e-6
    actual = A1(
        t=t,
        tau=t,
        x_t=shape.x(t),
        x_tau=shape.x(t),
        dx_tau=shape.dx(t),
        ddx_tau=shape.ddx(t),
        eps=xp.inf,
    )
    numerical = A1(
        t=t,
        tau=tau,
        x_t=shape.x(t),
        x_tau=shape.x(tau),
        dx_tau=shape.dx(tau),
        ddx_tau=shape.ddx(tau),
    )
    assert float(actual[0]) == pytest.approx(float(numerical[0]), rel=1e-3)
    assert float(actual[0]) == pytest.approx(-0.5, rel=1e-6)


@pytest.mark.parametrize("t_start_factor", [0, 0.5])
@pytest.mark.parametrize("n", [8, 128])
@pytest.mark.parametrize("tau_value", [0, 0.11])
@pytest.mark.parametrize("kind", ["slp", "dlp"])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_circle_case_matches_theorem(
    xp: Any,
    device: Any,
    dtype: Any,
    kind: Literal["slp", "dlp"],
    m: int,
    n: int,
    tau_value: float,
    t_start_factor: float,
) -> None:
    t, w = trapezoidal_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    t, r = shift_quadrature_singularity(log_cot_power_quadrature, tau_value)(
        n, 0, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    tau = xp.asarray(tau_value, device=device, dtype=dtype)
    shape, k = CircleShape(1), xp.asarray(1, device=device, dtype=dtype)
    values = {
        "t": t,
        "tau": tau,
        "k": k,
        "x_t": shape.x(t)[..., 0, :],
        "x_tau": shape.x(tau)[..., 0, :],
        "dx_tau": shape.dx(tau)[..., 0, :],
    }
    if kind == "slp":
        log_coeff, analytic = slp_kernel_split(**values)
    else:
        log_coeff, analytic = dlp_kernel_split(**values, ddx_tau=shape.ddx(tau)[..., 0, :])
    actual = xp.sum(xp.exp(1j * m * t) * (r * log_coeff + w * analytic))
    h = scipy_hankel1(m, k)
    j = scipy_jv(m, k, 0 if kind == "slp" else 1)
    expected = (1j * math.pi / 2) * h * j * xp.exp(1j * m * tau)
    if kind == "dlp":
        expected -= 0.5 * xp.exp(1j * m * tau)
    assert complex(actual) == pytest.approx(complex(expected), rel=5e-6)


@pytest.mark.parametrize("n", [8, 128])
@pytest.mark.parametrize("kind", ["slp", "dlp"])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_circle_sol_matches_theorem(
    xp: Any, device: Any, dtype: Any, kind: str, m: int, n: int
) -> None:
    shape, k = CircleShape(1), xp.asarray(1, device=device, dtype=dtype)
    h = scipy_hankel1(m, k)
    j = scipy_jv(m, k, 0 if kind == "slp" else 1)

    def coefficient(t: Array) -> Array:
        return xp.ones_like(t)[..., None] * (0 if kind == "slp" else 0.5)

    def rhs(t: Array) -> Array:
        return ((1j * math.pi / 2) * h * j * xp.exp(1j * m * t))[..., None]

    def kernel(t: Array, tau: Array, part: int) -> Array:
        values = {
            "t": t,
            "tau": tau,
            "k": k,
            "x_t": shape.x(t)[..., 0, :],
            "x_tau": shape.x(tau)[..., 0, :],
            "dx_tau": shape.dx(tau)[..., 0, :],
        }
        if kind == "slp":
            result = slp_kernel_split(**values)[part]
        else:
            result = dlp_kernel_split(**values, ddx_tau=shape.ddx(tau)[..., 0, :])[part]
        return result[..., None, None]

    kernels = {
        (QuadratureType.NO_SINGULARITY, 0): lambda t, tau: kernel(t, tau, 1),
        (QuadratureType.LOG_COT_POWER, 0): lambda t, tau: kernel(t, tau, 0),
    }
    density = nystrom(coefficient, kernels, rhs, n=n, xp=xp, device=device, dtype=dtype)
    eval_points = xp.random.random_uniform(shape=(3,), device=device, dtype=dtype) * 2 * math.pi
    actual = density(eval_points)
    expected = xp.exp(1j * m * eval_points)[..., None]
    assert actual.shape == expected.shape
    assert xp.all(xpx.isclose(actual, expected))
