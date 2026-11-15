from __future__ import annotations

from collections.abc import Callable

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import NystromInterpolant, Shape, trapezoidal_quadrature

from ._adjoint import objective_derivative, solve_adjoint


def scattering_cross_section(far_field: Array, /) -> Array:
    r"""Integrate $|u^\infty|^2$ over a full turn of equally spaced samples (M, ...)."""
    if far_field.ndim == 0 or far_field.shape[0] == 0:
        raise ValueError("far_field must contain at least one direction")
    xp = array_namespace(far_field)
    return xp.sum(xp.abs(far_field) ** 2, axis=0) * (2 * xp.pi / far_field.shape[0])


def _field_adjoint(
    t: Array,
    observations: Array,
    field_gradient: Array,
    /,
    *,
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
    kernel: Callable[..., Array],
    h_shape: Shape | None,
) -> Array:
    r"""
    Apply a sampled field kernel's adjoint to an objective's field gradient.

    ``field_gradient`` has shape (M, ...batch) and includes the observation
    integration factor. ``observations`` has shape (M, 2). If ``h_shape`` is
    supplied, ``kernel`` is the corresponding shape-derivative kernel.
    """
    check_shapes("Q,MD,M*B,D", t, observations, field_gradient, (2,))
    if (
        t.ndim != 1
        or t.shape[0] == 0
        or observations.ndim != 2
        or observations.shape[0] == 0
        or observations.shape[-1] != 2
        or field_gradient.shape[0] != observations.shape[0]
    ):
        raise ValueError("expected nodes (Q,), observations (M, 2), and gradient (M, ...)")
    xp = array_namespace(t, observations, field_gradient, k, alpha, eta)
    shape_values = shape.x(t[0])
    if shape_values.shape[-2] == 0:
        raise ValueError("at least one boundary is required")
    h_values = h_shape.x(t[0]) if h_shape is not None else shape_values
    if h_values.shape[-2] != shape_values.shape[-2]:
        raise ValueError("shape and h_shape must have the same number of boundaries")
    k, alpha, eta, _, _, _ = xp.broadcast_arrays(
        k,
        alpha,
        eta,
        field_gradient[0],
        xp.zeros(shape_values.shape[:-2], device=t.device, dtype=t.dtype),
        xp.zeros(h_values.shape[:-2], device=t.device, dtype=t.dtype),
    )
    parameter = t[(...,) + (None,) * k.ndim]
    points = observations[(None,) * (k.ndim + 1) + (slice(None), None, slice(None))]
    geometry = {
        "shape_x": shape.x(parameter)[..., None, :, :],
        "shape_dx": shape.dx(parameter)[..., None, :, :],
    }
    if h_shape is not None:
        geometry.update({
            "h": h_shape.x(parameter)[..., None, :, :],
            "dh": h_shape.dx(parameter)[..., None, :, :],
        })
    values = kernel(
        points,
        **geometry,
        k=k[..., None, None],
        alpha=alpha[..., None, None],
        eta=eta[..., None, None],
    )
    gradient = xp.moveaxis(field_gradient, 0, -1)
    return xp.sum(gradient[..., None] * xp.conj(values), axis=-2)


def field_objective_derivative(
    density: NystromInterpolant,
    observations: Array,
    field_gradient: Array,
    /,
    *,
    shape: Shape,
    incident_gradient: Array,
    k: Array,
    alpha: Array,
    eta: Array,
    n_solve: int,
    n_quad: int,
    kernel: Callable[..., Array],
    derivative_kernel: Callable[..., Array],
) -> tuple[NystromInterpolant, Callable[[Shape], Array]]:
    r"""
    Solve an adjoint and return directional derivatives for a field objective.

    ``observations`` has shape (M, 2). ``field_gradient`` has shape
    (M, ...batch) and contains $\partial J/\partial\mathrm{Re}\,u_m +
    i\partial J/\partial\mathrm{Im}\,u_m$, including observation quadrature
    factors. For the cross section this is $(4\pi/M)u^\infty$.

    ``incident_gradient`` is the incident field's spatial gradient, sampled
    at the ``n_quad`` nodes with shape (2*n_quad-1, ...batch, C, 2). ``kernel``
    and ``derivative_kernel`` evaluate the field kernel and its shape
    derivative. The returned pair is the adjoint density and a callable that
    evaluates the objective derivative for a perturbation ``Shape``.
    """
    xp = array_namespace(observations, field_gradient, incident_gradient, k, alpha, eta)
    dtype = xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta))
    t_solve, _ = trapezoidal_quadrature(n_solve, xp=xp, device=k.device, dtype=dtype)
    t, _ = trapezoidal_quadrature(n_quad, xp=xp, device=k.device, dtype=dtype)
    count = shape.x(t[0]).shape[-2]
    if (
        incident_gradient.ndim < 3
        or incident_gradient.shape[0] != t.shape[0]
        or incident_gradient.shape[-2:] != (count, 2)
    ):
        raise ValueError("incident_gradient must have shape (Q, ...batch, C, 2)")
    t_field, w = trapezoidal_quadrature(
        n_quad, xp=xp, device=observations.device, dtype=xp.result_type(observations, xp.real(k))
    )
    phi_field = density(t_field)
    if field_gradient.ndim == 0:
        raise ValueError("field_gradient must have a leading observation axis")
    batch, _, _, _, _ = xp.broadcast_arrays(
        k, alpha, eta, incident_gradient[0, ..., 0, 0], phi_field[0, ..., 0]
    )
    missing = max(0, batch.ndim - (field_gradient.ndim - 1))
    field_gradient = field_gradient[(slice(None),) + (None,) * missing + (...,)]
    field_gradient, _ = xp.broadcast_arrays(field_gradient, batch[None, ...])
    grad_phi_j = _field_adjoint(
        t_solve,
        observations,
        field_gradient,
        shape=shape,
        k=k,
        alpha=alpha,
        eta=eta,
        kernel=kernel,
        h_shape=None,
    )
    adjoint = solve_adjoint(
        shape=shape, k=k, alpha=alpha, eta=eta, n=n_solve, grad_phi_j=grad_phi_j
    )

    def derivative(h_shape: Shape, /) -> Array:
        direct = _field_adjoint(
            t_field,
            observations,
            field_gradient,
            shape=shape,
            k=k,
            alpha=alpha,
            eta=eta,
            kernel=derivative_kernel,
            h_shape=h_shape,
        )
        phi = phi_field[(slice(None),) + (None,) * (direct.ndim - phi_field.ndim) + (...,)]
        dr_j = xp.real(xp.sum(direct * xp.conj(phi), axis=(0, -1)) * w[0])
        h_t = h_shape.x(t[(...,) + (None,) * (direct.ndim - 2)])
        grad = incident_gradient[
            (slice(None),) + (None,) * (h_t.ndim - incident_gradient.ndim) + (...,)
        ]
        dr_g = -xp.sum(grad * h_t, axis=-1)
        return objective_derivative(
            k=k,
            shape=shape,
            alpha=alpha,
            eta=eta,
            n=n_quad,
            phi=density,
            adjoint=adjoint,
            dr_j=dr_j,
            dr_g=dr_g,
            h_shape=h_shape,
        )

    return adjoint, derivative
