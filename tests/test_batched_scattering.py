from itertools import product
from typing import Any

import pytest
from array_api.latest import Array, ArrayNamespace
from ie_circle import CircleShape, trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import (
    far_field,
    near_field,
    plot_near_field_prepare,
    scattering_dirichlet,
)
from biem_helmholtz_2d._adjoint import objective_derivative, solve_adjoint
from biem_helmholtz_2d._incident import plane_wave
from biem_helmholtz_2d._objective import grad_phi_abs2_scattered_field
from biem_helmholtz_2d._potential_inner_derivative import (
    dlp_shape_derivative_kernel,
    slp_shape_derivative_kernel,
)


class _Circle(CircleShape):
    def __init__(self, center: Array, /) -> None:
        super().__init__(1)
        object.__setattr__(self, "center", center)

    def x(self, t: Array, /) -> Array:
        return super().x(t) + self.center


@pytest.mark.parametrize("n_scatterers", [1, 2])
@pytest.mark.parametrize("batch_kind", ["eta", "mixed"])
def test_batched_scattering_and_adjoint(
    xp: ArrayNamespace, dtype: Any, device: Any, n_scatterers: int, batch_kind: str
) -> None:
    centers = xp.asarray([[0, 0], [5, 0]], dtype=dtype, device=device)
    shape = [_Circle(centers[i]) for i in range(n_scatterers)]
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
        n_plot=8,
        xlim=(-3, 7),
        ylim=(-4, 4),
    )
    assert all(
        entry["data"].shape == (8, 8, *batch_shape)
        for components in plotted.values()
        for entry in components.values()
    )
    target = xp.asarray(0, dtype=xp.result_type(dtype, 1j), device=device)

    def density_gradient(field: Array, k: Array, alpha: Array, eta: Array, /) -> Any:
        gradients = [
            grad_phi_abs2_scattered_field(
                point, field, shape=s, k=k, alpha=alpha, eta=eta, target=target
            )
            for s in shape
        ]

        def grad(parameter: Array, /) -> Array:
            parameter = parameter[(...,) + (None,) * k.ndim]
            return xp.stack([g(parameter) for g in gradients], axis=-1)

        return grad

    adjoint = solve_adjoint(
        k=k_input,
        shape=shape,
        alpha=alpha_input,
        eta=eta_input,
        n=n,
        grad_phi_j=density_gradient(field, k, alpha, eta),
    )
    perturbations = [CircleShape(1 if i == 0 else 0) for i in range(n_scatterers)]
    tau = t[(...,) + (None,) * len(batch_shape)]
    dr_g = xp.stack(
        [
            -1j * k * incident(s.x(tau)) * h.x(tau)[..., 0]
            for s, h in zip(shape, perturbations, strict=True)
        ],
        axis=-1,
    )
    sl = slp_shape_derivative_kernel(
        point,
        shape_x=shape[0].x,
        shape_dx=shape[0].dx,
        h=perturbations[0].x,
        dh=perturbations[0].dx,
        k=k,
        tau=tau,
    )
    dl = dlp_shape_derivative_kernel(
        point,
        shape_x=shape[0].x,
        shape_dx=shape[0].dx,
        h=perturbations[0].x,
        dh=perturbations[0].dx,
        k=k,
        tau=tau,
    )
    weights = w[(...,) + (None,) * len(batch_shape)]
    dfield = xp.sum((alpha * dl - 1j * eta * sl) * phi(t)[..., 0] * weights, axis=0)
    dr_j = 2 * xp.real(xp.conj(field) * dfield)
    derivative = objective_derivative(
        k=k_input,
        shape=shape,
        alpha=alpha_input,
        eta=eta_input,
        n=n,
        phi=phi,
        adjoint=adjoint,
        dr_j=dr_j,
        dr_g=dr_g,
        h_shape=perturbations,
    )
    assert derivative.shape == batch_shape

    for index in product(*(range(size) for size in batch_shape)):
        k_single, alpha_single, eta_single = k[index], alpha[index], eta[index]
        single_phi = scattering_dirichlet(
            k=k_single,
            shape=shape,
            incident_field=plane_wave(k_single, direction),
            alpha=alpha_single,
            eta=eta_single,
            n=n,
        )
        single_field = near_field(
            single_phi, point, k=k_single, shape=shape, alpha=alpha_single, eta=eta_single, n=n
        )
        single_adjoint = solve_adjoint(
            k=k_single,
            shape=shape,
            alpha=alpha_single,
            eta=eta_single,
            n=n,
            grad_phi_j=density_gradient(single_field, k_single, alpha_single, eta_single),
        )
        single_derivative = objective_derivative(
            k=k_single,
            shape=shape,
            alpha=alpha_single,
            eta=eta_single,
            n=n,
            phi=single_phi,
            adjoint=single_adjoint,
            dr_j=dr_j[index],
            dr_g=dr_g[slice(None), *index, slice(None)],
            h_shape=perturbations,
        )
        index_t = (slice(None), *index)
        tolerance = 500 * xp.finfo(dtype).eps
        assert bool(xp.all(xp.abs(phi(t)[index_t] - single_phi(t)) < tolerance))
        assert bool(xp.all(xp.abs(adjoint(t)[index_t] - single_adjoint(t)) < tolerance))
        assert bool(
            xp.all(
                xp.abs(
                    near[index_t]
                    - near_field(
                        single_phi,
                        points,
                        k=k_single,
                        shape=shape,
                        alpha=alpha_single,
                        eta=eta_single,
                        n=n,
                    )
                )
                < tolerance
            )
        )
        assert bool(
            xp.all(
                xp.abs(
                    far[index_t]
                    - far_field(
                        single_phi,
                        directions,
                        k=k_single,
                        shape=shape,
                        alpha=alpha_single,
                        eta=eta_single,
                        n=n,
                    )
                )
                < tolerance
            )
        )
        assert float(xp.abs(derivative[index] - single_derivative)) < tolerance
