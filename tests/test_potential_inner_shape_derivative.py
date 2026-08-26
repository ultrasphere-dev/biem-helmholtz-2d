from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from array_api.latest import ArrayNamespace
from ie_circle import Shape

from biem_helmholtz_2d import _potential_inner, _potential_inner_derivative


@pytest.mark.parametrize("k", [1, 2.5])
@pytest.mark.parametrize("epsilon", [1e-6])
def test_slp_shape_derivative_numerical(
    xp: ArrayNamespace,
    shape: Shape,
    shape_h: Shape,
    shape_central_difference: Callable[[float], tuple[Shape, Shape]],
    epsilon: float,
    k: float,
    device: Any,
    dtype: Any,
) -> None:
    n = 8
    x = xp.asarray([[3, 3]], device=device, dtype=dtype)
    func = xp.ones_like

    da = _potential_inner_derivative.slp_shape_derivative(
        x,
        func,
        shape_x=shape.x,
        shape_dx=shape.dx,
        h=shape_h.x,
        dh=shape_h.dx,
        k=k,
        n=n,
    )

    shape_p, shape_m = shape_central_difference(epsilon)

    p = _potential_inner.slp(x, func, shape_x=shape_p.x, shape_dx=shape_p.dx, k=k, n=n)
    m = _potential_inner.slp(x, func, shape_x=shape_m.x, shape_dx=shape_m.dx, k=k, n=n)

    num = (p - m) / (2 * epsilon)

    print(da, num)
    assert xp.all(xp.abs(da - num) < 1e-6 * xp.abs(da) + 1e-6), "SLP shape derivative mismatch"


@pytest.mark.parametrize("k", [1, 2.5])
@pytest.mark.parametrize("epsilon", [1e-6])
def test_dlp_shape_derivative_numerical(
    xp: ArrayNamespace,
    shape: Shape,
    shape_h: Shape,
    shape_central_difference: Callable[[float], tuple[Shape, Shape]],
    epsilon: float,
    k: float,
    device: Any,
    dtype: Any,
) -> None:
    n = 8
    x = xp.asarray([[3, 3]], device=device, dtype=dtype)
    func = xp.ones_like

    da = _potential_inner_derivative.dlp_shape_derivative(
        x,
        func,
        shape_x=shape.x,
        shape_dx=shape.dx,
        h=shape_h.x,
        dh=shape_h.dx,
        k=k,
        n=n,
    )

    shape_p, shape_m = shape_central_difference(epsilon)

    p = _potential_inner.dlp(x, func, shape_x=shape_p.x, shape_dx=shape_p.dx, k=k, n=n)
    m = _potential_inner.dlp(x, func, shape_x=shape_m.x, shape_dx=shape_m.dx, k=k, n=n)

    num = (p - m) / (2 * epsilon)

    print(da, num)
    assert xp.all(xp.abs(da - num) < 1e-6 * xp.abs(da) + 1e-6), "DLP shape derivative mismatch"
