from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import NystromInterpolant, Shape, trapezoidal_quadrature

from ._acoustic import far_field, near_field
from ._adjoint import objective_derivative, solve_adjoint
from ._potential_far import kernel as far_kernel
from ._potential_far_derivative import kernel as far_derivative_kernel
from ._potential_inner import kernel as near_kernel
from ._potential_inner_derivative import kernel as near_derivative_kernel

FieldTerm = tuple[Array, Array, Callable[..., Array], Callable[..., Array]]


def scattering_cross_section(far_field: Array, /) -> Array:
    r"""Integrate $|u^\infty|^2$ over a full turn of equally spaced samples (M, ...)."""
    if far_field.ndim == 0 or far_field.shape[0] == 0:
        raise ValueError("far_field must contain at least one direction")
    xp = array_namespace(far_field)
    return xp.sum(xp.abs(far_field) ** 2, axis=0) * (2 * xp.pi / far_field.shape[0])


def field_objective(
    density: NystromInterpolant,
    /,
    *,
    target: Array,
    weight_target: float,
    include_cross_section: bool,
    shape: Shape,
    incident_field: Callable[[Array], Array],
    k: Array,
    alpha: Array,
    eta: Array,
    n_quad: int,
    desired_total_field: complex = 0j,
) -> tuple[Array, list[FieldTerm]]:
    r"""
    Evaluate point-field loss and optionally the scattering cross section.

    ``target`` contains the observation points (P, 2). The point term is
    ``weight_target`` times the mean squared total-field error. When
    ``include_cross_section`` is true, the cross section is added and the
    desired field must be zero. The returned field terms can be passed to
    :func:`field_objective_derivative` to compute one adjoint for the sum.
    """
    xp = array_namespace(target, k, alpha, eta)
    if target.ndim != 2 or target.shape[0] == 0 or target.shape[1] != 2:
        raise ValueError("target must have shape (n_points, 2) with n_points > 0")
    if not math.isfinite(weight_target) or weight_target < 0:
        raise ValueError("weight_target must be finite and nonnegative")
    if include_cross_section and desired_total_field != 0:
        raise ValueError("the cloaking objective minimizes the total field itself")
    if not xp.isdtype(target.dtype, ("real floating", "integral")):
        raise ValueError("target points must be real")
    if not bool(xp.all(xp.isfinite(target))):
        raise ValueError("target points must be finite")
    scattered = near_field(density, target, shape=shape, k=k, alpha=alpha, eta=eta, n=n_quad)
    points = target[(slice(None),) + (None,) * (scattered.ndim - 1) + (slice(None),)]
    residual = scattered + incident_field(points) - desired_total_field
    value = weight_target * xp.sum(xp.abs(residual) ** 2, axis=0) / target.shape[0]
    terms: list[FieldTerm] = [
        (
            target,
            (2 * weight_target / target.shape[0]) * residual,
            near_kernel,
            near_derivative_kernel,
        )
    ]
    if include_cross_section:
        theta, _ = trapezoidal_quadrature(
            n_quad, xp=xp, device=target.device, dtype=xp.result_type(target, xp.real(k))
        )
        directions = xp.stack([xp.cos(theta), xp.sin(theta)], axis=-1)
        field = far_field(density, directions, shape=shape, k=k, alpha=alpha, eta=eta, n=n_quad)
        value = value + scattering_cross_section(field)
        terms.append((
            directions,
            (4 * xp.pi / directions.shape[0]) * field,
            far_kernel,
            far_derivative_kernel,
        ))
    return value, terms


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
    terms: Sequence[FieldTerm],
    /,
    *,
    shape: Shape,
    incident_gradient: Array,
    k: Array,
    alpha: Array,
    eta: Array,
    n_solve: int,
    n_quad: int,
) -> tuple[NystromInterpolant, Callable[[Shape], Array]]:
    r"""
    Solve one adjoint for a sum of field objectives and return its shape derivative.

    Each term contains observations (M, 2), field gradients (M, ...batch),
    the sampled field kernel, and its shape-derivative kernel. Gradients include
    observation integration factors; for the cross section this is
    $(4\pi/M)u^\infty$. ``incident_gradient`` has shape
    (2*n_quad-1, ...batch, C, 2).
    """
    if not terms:
        raise ValueError("at least one field term is required")
    xp = array_namespace(incident_gradient, k, alpha, eta, *(v for term in terms for v in term[:2]))
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
        n_quad,
        xp=xp,
        device=k.device,
        dtype=xp.result_type(xp.real(k), *(term[0] for term in terms)),
    )
    phi_field = density(t_field)
    batch, _, _, _, _ = xp.broadcast_arrays(
        k, alpha, eta, incident_gradient[0, ..., 0, 0], phi_field[0, ..., 0]
    )
    components = []
    for observations, field_gradient, kernel, derivative_kernel in terms:
        if field_gradient.ndim == 0:
            raise ValueError("field_gradient must have a leading observation axis")
        missing = max(0, batch.ndim - (field_gradient.ndim - 1))
        field_gradient = field_gradient[(slice(None),) + (None,) * missing + (...,)]
        field_gradient, _ = xp.broadcast_arrays(field_gradient, batch[None, ...])
        components.append((observations, field_gradient, kernel, derivative_kernel))
    grad_phi_j = sum(
        _field_adjoint(
            t_solve,
            observations,
            gradient,
            shape=shape,
            k=k,
            alpha=alpha,
            eta=eta,
            kernel=kernel,
            h_shape=None,
        )
        for observations, gradient, kernel, _ in components
    )
    adjoint = solve_adjoint(
        shape=shape, k=k, alpha=alpha, eta=eta, n=n_solve, grad_phi_j=grad_phi_j
    )

    def derivative(h_shape: Shape, /) -> Array:
        direct = sum(
            _field_adjoint(
                t_field,
                observations,
                gradient,
                shape=shape,
                k=k,
                alpha=alpha,
                eta=eta,
                kernel=derivative_kernel,
                h_shape=h_shape,
            )
            for observations, gradient, _, derivative_kernel in components
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
