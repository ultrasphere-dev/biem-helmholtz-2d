from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pytest
from array_api.latest import ArrayNamespace
from ie_circle import Shape, trapezoidal_quadrature

from biem_helmholtz_2d import _potential_inner, _potential_inner_derivative


@pytest.mark.parametrize("kind", ["single", "double"])
@pytest.mark.parametrize("k", [1, 2.5])
def test_shape_derivative_numerical(
    xp: ArrayNamespace,
    shape: Shape,
    shape_h: Shape,
    shape_central_difference: Callable[[float], tuple[Shape, Shape]],
    kind: Literal["single", "double"],
    k: float,
    device: Any,
    dtype: Any,
) -> None:
    epsilon = 1e-6
    x = xp.asarray([[3, 3]], device=device, dtype=dtype)
    wave_number = xp.asarray(k, device=device, dtype=dtype)
    t, _ = trapezoidal_quadrature(8, xp=xp, device=device, dtype=dtype)
    derivative = _potential_inner_derivative.potential_shape_derivative(
        x,
        xp.ones_like(t),
        kind=kind,
        shape_x=shape.x(t)[..., 0, :],
        shape_dx=shape.dx(t)[..., 0, :],
        h=shape_h.x(t)[..., 0, :],
        dh=shape_h.dx(t)[..., 0, :],
        k=wave_number,
    )
    shape_p, shape_m = shape_central_difference(epsilon)
    plus, minus = (
        _potential_inner.potential(
            x,
            xp.ones_like(t),
            kind=kind,
            shape_x=boundary.x(t)[..., 0, :],
            shape_dx=boundary.dx(t)[..., 0, :],
            k=wave_number,
        )
        for boundary in (shape_p, shape_m)
    )
    central = (plus - minus) / (2 * epsilon)
    assert xp.all(xp.abs(derivative - central) < 1e-6 * xp.abs(derivative) + 1e-6)
