from typing import Any

import pytest
from array_api.latest import Array
from attrs import frozen
from ie_circle import CircleShape, Shape, trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._adjoint import solve_adjoint
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import (
    point_field_adjoint_derivative,
    point_field_density_gradient,
    point_field_shape_derivative,
)


@frozen
class _TestShape:
    base: Shape
    perturbation: Shape
    amplitude: Array
    center: Array

    def x(self, t: Array, /) -> Array:
        return self.base.x(t) + self.amplitude[..., None] * self.perturbation.x(t) + self.center

    def dx(self, t: Array, /) -> Array:
        return self.base.dx(t) + self.amplitude[..., None] * self.perturbation.dx(t)

    def ddx(self, t: Array, /) -> Array:
        return self.base.ddx(t) + self.amplitude[..., None] * self.perturbation.ddx(t)


@pytest.mark.parametrize(
    (
        "n_scatterers",
        "moving",
        "quadrature_offset",
        "alpha_value",
        "eta_value",
        "target_value",
        "objective",
        "motion",
    ),
    [
        (1, 0, 0, 1, 1, 0, "squared", "shape"),
        (2, 0, 0, 1, 1, 0, "squared", "shape"),
        (2, 1, 4, 1 + 0.1j, 0.8 - 0.2j, 0.2 + 0.1j, "squared", "shape"),
        (3, 2, 0, 1, 1, 0, "real", "shape"),
        (2, 0, 0, 1, 1, 0, "squared", "translation"),
    ],
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
    objective: str,
    motion: str,
) -> None:
    """Compare the adjoint with finite differences for coupled boundaries."""
    n, step = 32, 1e-5
    k = xp.asarray(1, device=device, dtype=dtype)
    alpha = xp.asarray(alpha_value, device=device, dtype=xp.result_type(dtype, alpha_value))
    eta = xp.asarray(eta_value, device=device, dtype=xp.result_type(dtype, eta_value))
    target = xp.asarray(target_value, device=device, dtype=xp.result_type(dtype, target_value))
    point = xp.asarray([3, 3], device=device, dtype=dtype)
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident, incident_grad = plane_wave(k, direction), plane_wave_grad(k, direction)
    centers = xp.asarray([[0, 0], [6, 0], [12, 0]], device=device, dtype=dtype)[:n_scatterers]
    moving_mask = xp.astype(xp.arange(n_scatterers, device=device) == moving, dtype)
    h_amplitude = moving_mask if motion == "shape" else xp.zeros_like(moving_mask)
    h_centers = (
        moving_mask[:, None] * xp.asarray([0.3, 0.4], dtype=dtype, device=device)
        if motion == "translation"
        else xp.zeros_like(centers)
    )
    base_shape = shape
    shape = _TestShape(base_shape, shape_h, xp.zeros_like(moving_mask), centers)
    h_shape = _TestShape(CircleShape(0), shape_h, h_amplitude, h_centers)
    plus_shape = _TestShape(base_shape, shape_h, step * h_amplitude, centers + step * h_centers)
    minus_shape = _TestShape(base_shape, shape_h, -step * h_amplitude, centers - step * h_centers)
    quad = n + quadrature_offset

    def field_for(boundary: Shape, /) -> Any:
        density = scattering_dirichlet(
            k=k, shape=boundary, incident_field=incident, alpha=alpha, eta=eta, n=n
        )
        return near_field(density, point, shape=boundary, k=k, alpha=alpha, eta=eta, n=quad)

    density = scattering_dirichlet(
        k=k, shape=shape, incident_field=incident, alpha=alpha, eta=eta, n=n
    )
    field = near_field(density, point, shape=shape, k=k, alpha=alpha, eta=eta, n=quad)
    field_gradient = 2 * (field - target) if objective == "squared" else xp.ones_like(field)
    adjoint = solve_adjoint(
        k=k,
        shape=shape,
        alpha=alpha,
        eta=eta,
        n=n,
        grad_phi_j=point_field_density_gradient(
            point, field_gradient, shape=shape, k=k, alpha=alpha, eta=eta
        ),
    )
    t, _ = trapezoidal_quadrature(quad, xp=xp, device=device, dtype=dtype)
    actual = point_field_adjoint_derivative(
        density,
        adjoint,
        point=point,
        field_gradient=field_gradient,
        incident_gradient=incident_grad(shape.x(t)),
        shape=shape,
        h_shape=h_shape,
        k=k,
        alpha=alpha,
        eta=eta,
        n=quad,
    )
    plus, minus = field_for(plus_shape), field_for(minus_shape)
    if objective == "squared":
        expected = (xp.abs(plus - target) ** 2 - xp.abs(minus - target) ** 2) / (2 * step)
    else:
        expected = xp.real(plus - minus) / (2 * step)
    assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-8)
    if n_scatterers == 1:
        point32 = xp.astype(point, xp.float32)
        direct = point_field_shape_derivative(
            density, point32, shape=shape, h_shape=h_shape, k=k, alpha=alpha, eta=eta, n=36
        )
        plus = near_field(
            density,
            point32,
            shape=plus_shape,
            k=k,
            alpha=alpha,
            eta=eta,
            n=36,
        )
        minus = near_field(
            density,
            point32,
            shape=minus_shape,
            k=k,
            alpha=alpha,
            eta=eta,
            n=36,
        )
        assert float(xp.abs(direct - (plus - minus) / (2 * step))) < 2e-8
