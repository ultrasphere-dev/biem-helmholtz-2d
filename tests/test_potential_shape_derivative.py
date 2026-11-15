from collections.abc import Callable
from typing import Any

import pytest
from array_api.latest import ArrayNamespace
from ie_circle import Shape

from biem_helmholtz_2d._potential import dlp_kernel_split, slp_kernel_split
from biem_helmholtz_2d._potential_shape_derivative import (
    dlp_shape_derivative_split,
    slp_shape_derivative_split,
)


@pytest.mark.parametrize("kind", ["single", "double"])
@pytest.mark.parametrize("t_equals_tau", [True, False])
@pytest.mark.parametrize("k", [1, 2.5])
def test_shape_derivative_numerical(
    xp: ArrayNamespace,
    shape: Shape,
    shape_h: Shape,
    shape_central_difference: Callable[[float], tuple[Shape, Shape]],
    kind: str,
    k: float,
    device: Any,
    dtype: Any,
    t_equals_tau: bool,
) -> None:
    t = xp.asarray(0, device=device, dtype=dtype)
    tau = xp.asarray(0 if t_equals_tau else 1, device=device, dtype=dtype)
    k = xp.asarray(k, device=device, dtype=dtype)
    step = 1e-6

    def kernel(boundary: Shape, /) -> tuple[Any, Any]:
        values = {
            "t": t,
            "tau": tau,
            "k": k,
            "x_t": boundary.x(t),
            "x_tau": boundary.x(tau),
            "dx_tau": boundary.dx(tau),
        }
        if kind == "single":
            return slp_kernel_split(**values)
        return dlp_kernel_split(**values, ddx_tau=boundary.ddx(tau))

    values = {
        "t": t,
        "tau": tau,
        "k": k,
        "x_t": shape.x(t),
        "x_tau": shape.x(tau),
        "dx_tau": shape.dx(tau),
        "h_t": shape_h.x(t),
        "h_tau": shape_h.x(tau),
        "dh_tau": shape_h.dx(tau),
    }
    if kind == "single":
        actual = slp_shape_derivative_split(**values)
    else:
        actual = dlp_shape_derivative_split(
            **values, ddx_tau=shape.ddx(tau), ddh_tau=shape_h.ddx(tau)
        )
    plus, minus = shape_central_difference(step)
    for value, p, m in zip(actual, kernel(plus), kernel(minus), strict=True):
        expected = (p - m) / (2 * step)
        assert xp.all(xp.abs(value - expected) < 1e-6 * xp.abs(value) + 1e-6)
