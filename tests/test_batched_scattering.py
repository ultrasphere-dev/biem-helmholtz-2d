from itertools import product
from typing import Any

import pytest
from array_api.latest import ArrayNamespace
from ie_circle import trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import (
    far_field,
    near_field,
    plot_near_field_prepare,
    scattering_dirichlet,
)
from biem_helmholtz_2d._adjoint import objective_derivative, solve_adjoint
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import field_objective_derivative, scattering_cross_section
from biem_helmholtz_2d._potential_far import kernel as far_kernel
from biem_helmholtz_2d._potential_far_derivative import kernel as far_derivative_kernel
from biem_helmholtz_2d._potential_inner import kernel as near_kernel
from biem_helmholtz_2d._potential_inner_derivative import kernel as near_derivative_kernel
from biem_helmholtz_2d.optimization._shape import ParameterShape


@pytest.mark.parametrize("n_scatterers", [1, 2])
@pytest.mark.parametrize("batch_kind", ["eta", "mixed"])
@pytest.mark.parametrize("objective", ["point", "cross_section", "linear"])
def test_batched_scattering_and_adjoint(
    xp: ArrayNamespace,
    dtype: Any,
    device: Any,
    n_scatterers: int,
    batch_kind: str,
    objective: str,
) -> None:
    centers = xp.asarray([[0, 0], [5, 0]], dtype=dtype, device=device)[:n_scatterers]
    sin_coefs = xp.zeros((n_scatterers, 0), dtype=dtype, device=device)
    shape = ParameterShape(
        cos_coefs=xp.ones((n_scatterers, 1), dtype=dtype, device=device),
        sin_coefs=sin_coefs,
        center=centers,
    )
    perturbation = ParameterShape(
        cos_coefs=xp.astype(xp.arange(n_scatterers, device=device) == 0, dtype)[:, None],
        sin_coefs=sin_coefs,
        center=xp.zeros_like(centers),
    )
    direction = xp.asarray([1, 0], dtype=dtype, device=device)
    point = xp.asarray([2, 3], dtype=dtype, device=device)
    points = xp.asarray([[2, 3], [-3, 2]], dtype=dtype, device=device)
    directions = xp.asarray([[1, 0], [0, 1]], dtype=dtype, device=device)
    if batch_kind == "eta":
        k = xp.asarray(1, dtype=dtype, device=device)
        alpha = xp.asarray(1, dtype=dtype, device=device)
        eta = xp.asarray([0.7, 1.3], dtype=dtype, device=device)
    else:
        k = xp.asarray([[0.8, 1.1]], dtype=dtype, device=device)
        alpha = xp.asarray([[0.9], [1.2]], dtype=dtype, device=device)
        eta = xp.asarray([[0.7, 1.3], [1.1, 0.8]], dtype=dtype, device=device)
    k_input, alpha_input, eta_input = k, alpha, eta
    k, alpha, eta = xp.broadcast_arrays(k, alpha, eta)
    batch_shape = k.shape
    n = 6
    t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    incident = plane_wave(k_input, direction)
    phi = scattering_dirichlet(
        k=k_input, shape=shape, incident_field=incident, alpha=alpha_input, eta=eta_input, n=n
    )
    near = near_field(phi, points, k=k_input, shape=shape, alpha=alpha_input, eta=eta_input, n=n)
    far = far_field(phi, directions, k=k_input, shape=shape, alpha=alpha_input, eta=eta_input, n=n)
    scaled_far = far_field(
        phi, 2 * directions, k=k_input, shape=shape, alpha=alpha_input, eta=eta_input, n=n
    )
    assert bool(xp.all(xp.abs(far - scaled_far) < 500 * xp.finfo(dtype).eps))
    assert phi(t).shape == (2 * n - 1, *batch_shape, n_scatterers)
    assert near.shape == (2, *batch_shape)
    assert far.shape == (2, *batch_shape)
    field = near_field(phi, point, k=k_input, shape=shape, alpha=alpha_input, eta=eta_input, n=n)
    plotted = plot_near_field_prepare(
        phi,
        incident,
        shape=shape,
        k=k_input,
        alpha=alpha_input,
        eta=eta_input,
        n=n,
        n_plot=9,
        xlim=(-3, 7),
        ylim=(-4, 4),
    )
    assert all(
        entry["data"].shape == (9, 9, *batch_shape)
        for components in plotted.values()
        for entry in components.values()
    )
    plot_point = xp.asarray([7, 2], dtype=dtype, device=device)
    plot_field = near_field(
        phi, plot_point, k=k_input, shape=shape, alpha=alpha_input, eta=eta_input, n=n
    )
    assert bool(
        xp.all(
            xp.abs(plotted["uscat"]["re"]["data"][-1, 6] - xp.real(plot_field))
            < 500 * xp.finfo(dtype).eps
        )
    )
    assert bool(
        xp.all(
            xp.abs(
                plotted["utot"]["re"]["data"][-1, 6] - xp.real(plot_field + incident(plot_point))
            )
            < 500 * xp.finfo(dtype).eps
        )
    )
    tau = t[(...,) + (None,) * len(batch_shape)]
    boundary_points = xp.moveaxis(shape.x(tau), -2, 0)
    incident_gradient = xp.moveaxis(plane_wave_grad(k, direction)(boundary_points), 0, -2)
    dr_g = -xp.sum(incident_gradient * perturbation.x(tau), axis=-1)
    if objective == "cross_section":
        observations = xp.stack([xp.cos(t), xp.sin(t)], axis=-1)
        objective_field = far_field(phi, observations, k=k, shape=shape, alpha=alpha, eta=eta, n=n)
        field_gradient = 2 * objective_field * (2 * xp.pi / observations.shape[0])
        kernel, derivative_kernel = far_kernel, far_derivative_kernel
        assert scattering_cross_section(objective_field).shape == batch_shape
        constant_cross_section = scattering_cross_section(xp.ones_like(objective_field))
        assert bool(xp.all(xp.abs(constant_cross_section - 2 * xp.pi) < 500 * xp.finfo(dtype).eps))
    elif objective == "linear":
        observations = xp.concat([points, point[None, :]], axis=0)
        field_gradient = xp.ones(observations.shape[0], dtype=dtype, device=device)
        kernel, derivative_kernel = near_kernel, near_derivative_kernel
    else:
        observations = point[None, :]
        field_gradient = (2 * field)[None, ...]
        kernel, derivative_kernel = near_kernel, near_derivative_kernel
    adjoint, directional = field_objective_derivative(
        phi,
        observations,
        field_gradient,
        incident_gradient=incident_gradient,
        shape=shape,
        k=k_input,
        alpha=alpha_input,
        eta=eta_input,
        n_solve=n,
        n_quad=n,
        kernel=kernel,
        derivative_kernel=derivative_kernel,
    )
    derivative = directional(perturbation)
    assert derivative.shape == batch_shape
    if objective == "linear":
        field_gradient = xp.broadcast_to(
            field_gradient[(slice(None),) + (None,) * len(batch_shape)],
            (observations.shape[0], *batch_shape),
        )
    points_expanded = observations[
        (None,) * (len(batch_shape) + 1) + (slice(None), None, slice(None))
    ]
    kernel_h = derivative_kernel(
        points_expanded,
        shape_x=shape.x(tau)[..., None, :, :],
        shape_dx=shape.dx(tau)[..., None, :, :],
        h=perturbation.x(tau)[..., None, :, :],
        dh=perturbation.dx(tau)[..., None, :, :],
        k=k[..., None, None],
        alpha=alpha[..., None, None],
        eta=eta[..., None, None],
    )
    dfield = xp.sum(kernel_h * phi(t)[..., None, :], axis=(0, -1)) * w[0]
    dr_j = xp.real(xp.sum(xp.conj(xp.moveaxis(field_gradient, 0, -1)) * dfield, axis=-1))

    for index in product(*(range(size) for size in batch_shape)):
        ks, alphas, etas = k[index], alpha[index], eta[index]
        single_phi = scattering_dirichlet(
            k=ks, shape=shape, incident_field=plane_wave(ks, direction), alpha=alphas, eta=etas, n=n
        )
        single_kernel = kernel(
            observations[None, :, None, :],
            shape_x=shape.x(t)[:, None, :, :],
            shape_dx=shape.dx(t)[:, None, :, :],
            k=ks,
            alpha=alphas,
            eta=etas,
        )
        single_gradient = field_gradient[slice(None), *index]
        grad_phi_j = xp.sum(single_gradient[None, :, None] * xp.conj(single_kernel), axis=1)
        single_adjoint = solve_adjoint(
            k=ks,
            shape=shape,
            alpha=alphas,
            eta=etas,
            n=n,
            grad_phi_j=grad_phi_j,
        )
        single_derivative = objective_derivative(
            k=ks,
            shape=shape,
            alpha=alphas,
            eta=etas,
            n=n,
            phi=single_phi,
            adjoint=single_adjoint,
            dr_j=dr_j[index],
            dr_g=dr_g[slice(None), *index, slice(None)],
            h_shape=perturbation,
        )
        index_t = (slice(None), *index)
        tolerance = 500 * xp.finfo(dtype).eps
        assert bool(xp.all(xp.abs(phi(t)[index_t] - single_phi(t)) < tolerance))
        assert bool(xp.all(xp.abs(adjoint(t)[index_t] - single_adjoint(t)) < tolerance))
        assert bool(
            xp.all(
                xp.abs(
                    near[index_t]
                    - near_field(single_phi, points, k=ks, shape=shape, alpha=alphas, eta=etas, n=n)
                )
                < tolerance
            )
        )
        assert bool(
            xp.all(
                xp.abs(
                    far[index_t]
                    - far_field(
                        single_phi, directions, k=ks, shape=shape, alpha=alphas, eta=etas, n=n
                    )
                )
                < tolerance
            )
        )
        assert float(xp.abs(derivative[index] - single_derivative)) < tolerance
