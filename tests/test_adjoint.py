from typing import Any

import pytest
from array_api.latest import Array
from attrs import frozen
from ie_circle import CircleShape, Shape, trapezoidal_quadrature
from scipy.special import hankel1, jv

from biem_helmholtz_2d._acoustic import far_field, near_field, scattering_dirichlet
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import (
    field_objective,
    field_objective_derivative,
    scattering_cross_section,
)
from biem_helmholtz_2d._potential_far import kernel as far_kernel
from biem_helmholtz_2d._potential_far_derivative import kernel as far_derivative_kernel
from biem_helmholtz_2d._potential_inner import kernel as near_kernel
from biem_helmholtz_2d._potential_inner_derivative import kernel as near_derivative_kernel


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


@pytest.mark.parametrize(("n_scatterers", "quadrature_offset"), [(1, 0), (2, 4)])
def test_cloaking_derivative(
    xp: Any,
    shape: Shape,
    shape_h: Shape,
    device: Any,
    dtype: Any,
    n_scatterers: int,
    quadrature_offset: int,
) -> None:
    n, step, weight_target = 32, 1e-5, 2.5
    k, alpha, eta = (xp.asarray(v, dtype=dtype, device=device) for v in (1, 1, 1))
    target = xp.asarray([[3, 3], [-2, 3], [0, 3]], dtype=dtype, device=device)
    centers = xp.asarray([[0, 0], [6, 0]], dtype=dtype, device=device)[:n_scatterers]
    mask = xp.astype(xp.arange(n_scatterers, device=device) == n_scatterers - 1, dtype)
    zeros = xp.zeros_like(centers)
    boundary = _TestShape(shape, shape_h, xp.zeros_like(mask), centers)
    perturbation = _TestShape(CircleShape(0), shape_h, mask, zeros)
    incident = plane_wave(k, xp.asarray([1, 0], dtype=dtype, device=device))
    incident_grad = plane_wave_grad(k, xp.asarray([1, 0], dtype=dtype, device=device))
    quad = n + quadrature_offset
    t, _ = trapezoidal_quadrature(quad, xp=xp, device=device, dtype=dtype)

    def evaluate(boundary_shape: Shape, /) -> tuple[Any, Any, Any]:
        density = scattering_dirichlet(
            k=k, shape=boundary_shape, incident_field=incident, alpha=alpha, eta=eta, n=n
        )
        value, terms = field_objective(
            density,
            target=target,
            weight_target=weight_target,
            include_cross_section=True,
            shape=boundary_shape,
            incident_field=incident,
            k=k,
            alpha=alpha,
            eta=eta,
            n_quad=quad,
        )
        return density, value, terms

    density, value, terms = evaluate(boundary)
    directions = xp.stack([xp.cos(t), xp.sin(t)], axis=-1)
    far = far_field(density, directions, k=k, shape=boundary, alpha=alpha, eta=eta, n=quad)
    total = incident(target) + near_field(
        density, target, k=k, shape=boundary, alpha=alpha, eta=eta, n=quad
    )
    expected_value = scattering_cross_section(far) + weight_target * xp.sum(xp.abs(total) ** 2) / 3
    assert float(value) == pytest.approx(float(expected_value), rel=1e-13, abs=1e-13)
    _, derivative = field_objective_derivative(
        density,
        terms,
        shape=boundary,
        incident_gradient=incident_grad(boundary.x(t)),
        k=k,
        alpha=alpha,
        eta=eta,
        n_solve=n,
        n_quad=quad,
    )
    plus = _TestShape(shape, shape_h, step * mask, centers)
    minus = _TestShape(shape, shape_h, -step * mask, centers)
    finite_difference = (evaluate(plus)[1] - evaluate(minus)[1]) / (2 * step)
    assert float(derivative(perturbation)) == pytest.approx(
        float(finite_difference), rel=1e-5, abs=1e-8
    )


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
        (1, 0, 0, 1, 1, 0, "cross_section", "shape"),
        (2, 1, 4, 1 + 0.1j, 0.8 - 0.2j, 0, "cross_section", "shape"),
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
    t, _ = trapezoidal_quadrature(quad, xp=xp, device=device, dtype=dtype)
    directions = xp.stack([xp.cos(t), xp.sin(t)], axis=-1)
    observations = directions if objective == "cross_section" else point[None, :]
    kernel, derivative_kernel = (
        (far_kernel, far_derivative_kernel)
        if objective == "cross_section"
        else (near_kernel, near_derivative_kernel)
    )

    def field_for(boundary: Shape, /) -> Any:
        density = scattering_dirichlet(
            k=k, shape=boundary, incident_field=incident, alpha=alpha, eta=eta, n=n
        )
        evaluator = far_field if objective == "cross_section" else near_field
        return evaluator(density, observations, shape=boundary, k=k, alpha=alpha, eta=eta, n=quad)

    density = scattering_dirichlet(
        k=k, shape=shape, incident_field=incident, alpha=alpha, eta=eta, n=n
    )
    evaluator = far_field if objective == "cross_section" else near_field
    field = evaluator(density, observations, shape=shape, k=k, alpha=alpha, eta=eta, n=quad)
    if objective == "cross_section":
        field_gradient = 2 * field * (2 * xp.pi / directions.shape[0])
    elif objective == "squared":
        field_gradient = 2 * (field - target)
    else:
        field_gradient = xp.ones_like(field)
    _, directional = field_objective_derivative(
        density,
        [(observations, field_gradient, kernel, derivative_kernel)],
        incident_gradient=incident_grad(shape.x(t)),
        shape=shape,
        k=k,
        alpha=alpha,
        eta=eta,
        n_solve=n,
        n_quad=quad,
    )
    actual = directional(h_shape)
    plus, minus = field_for(plus_shape), field_for(minus_shape)
    if objective == "cross_section":
        expected = (scattering_cross_section(plus) - scattering_cross_section(minus)) / (2 * step)
        if n_scatterers == 1 and isinstance(base_shape, CircleShape):
            reference = 4 * sum(abs(jv(m, 1) / hankel1(m, 1)) ** 2 for m in range(-30, 31))
            assert float(scattering_cross_section(field)) == pytest.approx(reference, rel=1e-11)
    elif objective == "squared":
        expected = (xp.abs(plus[0] - target) ** 2 - xp.abs(minus[0] - target) ** 2) / (2 * step)
    else:
        expected = xp.real(plus[0] - minus[0]) / (2 * step)
    assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-8)
    if n_scatterers == 1:
        point32 = xp.astype(point, xp.float32)
        t_field, w_field = trapezoidal_quadrature(
            36, xp=xp, device=device, dtype=xp.result_type(point32, xp.real(k))
        )
        direct_kernel = near_derivative_kernel(
            point32,
            shape_x=shape.x(t_field),
            shape_dx=shape.dx(t_field),
            h=h_shape.x(t_field),
            dh=h_shape.dx(t_field),
            k=k,
            alpha=alpha,
            eta=eta,
        )
        direct = xp.sum(direct_kernel * density(t_field), axis=(0, -1)) * w_field[0]
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
