from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from array_api.latest import Array
from ie_circle import CircleShape, Shape

from biem_helmholtz_2d import adjoint_central_derivative
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad


@dataclass(frozen=True)
class ShiftedShape:
    shape: Shape
    center: Array

    def x(self, t: Array, /) -> Array:
        return self.shape.x(t) + self.center

    def dx(self, t: Array, /) -> Array:
        return self.shape.dx(t)

    def ddx(self, t: Array, /) -> Array:
        return self.shape.ddx(t)


@pytest.mark.parametrize(
    ("n_scatterers", "moving", "quadrature_offset", "alpha_value", "eta_value", "target_value"),
    [(1, 0, 0, 1, 1, 0), (2, 0, 0, 1, 1, 0), (2, 1, 4, 1 + 0.1j, 0.8 - 0.2j, 0.2 + 0.1j)],
)
def test_adjoint_central_derivative(
    xp: Any,
    shape: Shape,
    shape_h: Shape,
    device: Any,
    dtype: Any,
    n_scatterers: int,
    moving: int,
    quadrature_offset: int,
    alpha_value: complex,
    eta_value: complex,
    target_value: complex,
) -> None:
    r"""
    Compare adjoint-based shape derivative with central finite differences.

    Objective $J(x) = |u_{\mathrm{scat}}(x_0, x)|^2$ (scattered field at
    $x_0 = (3,3)$). Perturb one boundary by $h = \mathtt{shape\_h}$,
    with either one or two disjoint scatterers.
    """
    n = 32
    k_arr = xp.asarray(1, device=device, dtype=dtype)
    alpha = xp.asarray(alpha_value, device=device, dtype=xp.result_type(dtype, alpha_value))
    eta = xp.asarray(eta_value, device=device, dtype=xp.result_type(dtype, eta_value))
    x0 = xp.asarray([3, 3], device=device, dtype=dtype)
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident_field = plane_wave(k_arr, direction)
    centers = xp.asarray([[0, 0], [6, 0]], device=device, dtype=dtype)
    shapes = [ShiftedShape(shape, centers[i]) for i in range(n_scatterers)]
    perturbations = [shape_h if i == moving else CircleShape(0) for i in range(n_scatterers)]

    result = adjoint_central_derivative(
        shape=shapes,
        h_shape=perturbations,
        point=x0,
        target=xp.asarray(target_value, dtype=xp.result_type(dtype, target_value), device=device),
        k=k_arr,
        alpha=alpha,
        eta=eta,
        incident_field=incident_field,
        incident_field_grad=plane_wave_grad(k_arr, direction),
        n_solve=n,
        n_quad=n + quadrature_offset,
        steps=[1e-5],
    )
    assert float(result.adjoint) == pytest.approx(
        float(result.central_objective[0]), rel=1e-5, abs=1e-8
    )
