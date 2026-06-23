from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import sympy
from array_api.latest import ArrayNamespace
from ie_circle import SympyShape

from biem_helmholtz_2d import _potential, _potential_shape_derivative


def circle_shape(t: sympy.Symbol) -> tuple[sympy.Expr, sympy.Expr]:
    rho = 2.0
    return rho * sympy.cos(t), rho * sympy.sin(t)


def perturbation_shape(t: sympy.Symbol) -> tuple[sympy.Expr, sympy.Expr]:
    r = sympy.cos(2 * t)
    return r * sympy.cos(t), r * sympy.sin(t)


ShapeFunc = Callable[[sympy.Symbol], tuple[sympy.Expr, sympy.Expr]]


@pytest.mark.parametrize("t_equals_tau", [True, False])
@pytest.mark.parametrize("k", [1.0, 2.5])
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("func_h", [perturbation_shape])
@pytest.mark.parametrize("func_x", [circle_shape])
def test_slp_shape_derivative_numerical(
    xp: ArrayNamespace,
    func_x: ShapeFunc,
    func_h: ShapeFunc,
    epsilon: float,
    k: float,
    device: Any,
    dtype: Any,
    t_equals_tau: bool,
) -> None:
    t_sym = sympy.Symbol("t")
    shape_x = SympyShape(*func_x(t_sym), t_sym)
    shape_h = SympyShape(*func_h(t_sym), t_sym)

    t = xp.asarray(0.0, device=device, dtype=dtype)
    tau = xp.asarray(0.0 if t_equals_tau else 1.0, device=device, dtype=dtype)

    da_log, da_rem = _potential_shape_derivative.slp_shape_derivative_split(
        t=t,
        tau=tau,
        k=k,
        x=shape_x.x,
        dx=shape_x.dx,
        h=shape_h.x,
        dh=shape_h.dx,
    )

    def make_shape(eps):
        return SympyShape(
            shape_x.x_expr + eps * shape_h.x_expr,
            shape_x.y_expr + eps * shape_h.y_expr,
            shape_x.t_symbol,
        )

    shape_p = make_shape(epsilon)
    shape_m = make_shape(-epsilon)

    p_log, p_rem = _potential.slp_kernel_split(t=t, tau=tau, k=k, x=shape_p.x, dx=shape_p.dx)
    m_log, m_rem = _potential.slp_kernel_split(t=t, tau=tau, k=k, x=shape_m.x, dx=shape_m.dx)

    num_log = (p_log - m_log) / (2 * epsilon)
    num_rem = (p_rem - m_rem) / (2 * epsilon)

    # Check relative error
    # Use higher tolerance for finite difference approximation
    print(da_log, num_log)
    print(da_rem, num_rem)
    assert xp.all(xp.abs(da_log - num_log) < 1e-6 * xp.abs(da_log) + 1e-6), "SLP Log part mismatch"
    assert xp.all(xp.abs(da_rem - num_rem) < 1e-6 * xp.abs(da_rem) + 1e-6), (
        "SLP Remainder part mismatch"
    )


@pytest.mark.parametrize("t_equals_tau", [True, False])
@pytest.mark.parametrize("k", [1.0, 2.5])
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("func_h", [perturbation_shape])
@pytest.mark.parametrize("func_x", [circle_shape])
def test_dlp_shape_derivative_numerical(
    xp: ArrayNamespace,
    func_x: ShapeFunc,
    func_h: ShapeFunc,
    epsilon: float,
    k: float,
    device: Any,
    dtype: Any,
    t_equals_tau: bool,
) -> None:
    t_sym = sympy.Symbol("t")
    shape_x = SympyShape(*func_x(t_sym), t_sym)
    shape_h = SympyShape(*func_h(t_sym), t_sym)

    t = xp.asarray(0.0, device=device, dtype=dtype)
    tau = xp.asarray(0.0 if t_equals_tau else 1.0, device=device, dtype=dtype)

    da_log, da_rem = _potential_shape_derivative.dlp_shape_derivative_split(
        t=t,
        tau=tau,
        k=k,
        x=shape_x.x,
        dx=shape_x.dx,
        ddx=shape_x.ddx,
        h=shape_h.x,
        dh=shape_h.dx,
        ddh=shape_h.ddx,
    )

    def make_shape(eps):
        return SympyShape(
            shape_x.x_expr + eps * shape_h.x_expr,
            shape_x.y_expr + eps * shape_h.y_expr,
            shape_x.t_symbol,
        )

    shape_p = make_shape(epsilon)
    shape_m = make_shape(-epsilon)

    p_log, p_rem = _potential.dlp_kernel_split(
        t=t, tau=tau, k=k, x=shape_p.x, dx=shape_p.dx, ddx=shape_p.ddx
    )
    m_log, m_rem = _potential.dlp_kernel_split(
        t=t, tau=tau, k=k, x=shape_m.x, dx=shape_m.dx, ddx=shape_m.ddx
    )

    num_log = (p_log - m_log) / (2 * epsilon)
    num_rem = (p_rem - m_rem) / (2 * epsilon)

    print(da_log, num_log)
    print(da_rem, num_rem)
    assert xp.all(xp.abs(da_log - num_log) < 1e-6 * xp.abs(da_log) + 1e-6), "DLP Log part mismatch"
    assert xp.all(xp.abs(da_rem - num_rem) < 1e-6 * xp.abs(da_rem) + 1e-6), (
        "DLP Remainder part mismatch"
    )
