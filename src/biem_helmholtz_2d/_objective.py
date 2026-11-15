from __future__ import annotations

from collections.abc import Callable

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import NystromInterpolant, Shape, trapezoidal_quadrature

from ._adjoint import objective_derivative
from ._potential_inner import dlp_kernel, slp_kernel
from ._potential_inner_derivative import dlp_shape_derivative_kernel, slp_shape_derivative_kernel


def abs2_scattered_field(target: Array, /, *, field_value: Array) -> Array:
    r"""
    Objective function $j(\phi) = |u(x_0) - c|^2$.

    Parameters
    ----------
    target : Array
        Target value $c$ of shape (...,).
    field_value : Array
        Scattered field $u(x_0)$ of shape (...,).

    Returns
    -------
    Array
        Objective value $j(\phi)$ of shape (...,).

    """
    xp = array_namespace(target, field_value)
    return xp.abs(field_value - target) ** 2


def point_field_density_gradient(
    point: Array,
    field_gradient: Array,
    /,
    *,
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Callable[[Array], Array]:
    r"""
    Return the density gradient for a real objective depending on a point field.

    ``field_gradient`` is $\partial J/\partial\operatorname{Re}u +
    i\partial J/\partial\operatorname{Im}u$.
    """
    check_shapes("D,D", point, (2,), names="point,coordinate_dimension")
    if point.shape != (2,):
        raise ValueError("point must have shape (2,)")
    xp = array_namespace(point, field_gradient, k, alpha, eta)
    k, alpha, eta, field_gradient = xp.broadcast_arrays(k, alpha, eta, field_gradient)
    batch_ndim = k.ndim

    def gradient(t: Array, /) -> Array:
        parameter = t[(...,) + (None,) * batch_ndim]
        x_t, dx_t = shape.x(parameter), shape.dx(parameter)
        sl = slp_kernel(point, shape_x=x_t, shape_dx=dx_t, k=k[..., None])
        dl = dlp_kernel(point, shape_x=x_t, shape_dx=dx_t, k=k[..., None])
        return field_gradient[..., None] * xp.conj(alpha[..., None] * dl - 1j * eta[..., None] * sl)

    return gradient


def point_field_shape_derivative(
    density: NystromInterpolant,
    point: Array,
    /,
    *,
    shape: Shape,
    h_shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
    n: int,
) -> Array:
    """Return the point-field derivative with respect to shape at fixed density."""
    check_shapes("D,D", point, (2,), names="point,coordinate_dimension")
    if point.shape != (2,):
        raise ValueError("point must have shape (2,)")
    xp = array_namespace(point, k, alpha, eta)
    t, weights = trapezoidal_quadrature(
        n, xp=xp, device=point.device, dtype=xp.result_type(point, xp.real(k))
    )
    count = shape.x(t[0]).shape[-2]
    h_values = h_shape.x(t[0])
    if count == 0 or h_values.shape[-2] != count:
        raise ValueError("shape and h_shape must have the same nonzero length")
    values = density(t)
    check_shapes("Q*BC,QC", values, (t.shape[0], count), names="density,expected")
    if values.shape[0] != t.shape[0] or values.shape[-1] != count:
        raise ValueError("density must have one value per quadrature node and scatterer")
    h_batch = h_values.shape[:-2]
    k, alpha, eta, _, _ = xp.broadcast_arrays(
        k, alpha, eta, values[0, ..., 0], xp.zeros(h_batch, dtype=point.dtype, device=point.device)
    )
    parameter = t[(...,) + (None,) * k.ndim]
    x_t, dx_t = shape.x(parameter), shape.dx(parameter)
    h_t, dh_t = h_shape.x(parameter), h_shape.dx(parameter)
    sl = slp_shape_derivative_kernel(
        point, shape_x=x_t, shape_dx=dx_t, h=h_t, dh=dh_t, k=k[..., None]
    )
    dl = dlp_shape_derivative_kernel(
        point, shape_x=x_t, shape_dx=dx_t, h=h_t, dh=dh_t, k=k[..., None]
    )
    weights = weights[(slice(None),) + (None,) * (k.ndim + 1)]
    values = values[(slice(None),) + (None,) * (k.ndim - values.ndim + 2) + (...,)]
    return xp.sum(
        (alpha[..., None] * dl - 1j * eta[..., None] * sl) * values * weights, axis=(0, -1)
    )


def point_field_adjoint_derivative(
    density: NystromInterpolant,
    adjoint: NystromInterpolant,
    /,
    *,
    point: Array,
    field_gradient: Array,
    incident_gradient: Array,
    shape: Shape,
    h_shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
    n: int,
) -> Array:
    """
    Return an adjoint derivative for a real objective depending on a point field.

    ``incident_gradient`` contains the incident-field gradient at the ``n``
    integration nodes, with shape (Q, ...batch, C, 2).
    """
    xp = array_namespace(point, field_gradient, incident_gradient, k, alpha, eta)
    t, _ = trapezoidal_quadrature(
        n,
        xp=xp,
        device=point.device,
        dtype=xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta)),
    )
    count = shape.x(t[0]).shape[-2]
    h_values = h_shape.x(t[0])
    if h_values.shape[-2] != count:
        raise ValueError("shape and h_shape must have the same number of boundaries")
    if (
        incident_gradient.ndim < 3
        or incident_gradient.shape[0] != t.shape[0]
        or incident_gradient.shape[-2:] != (count, 2)
    ):
        raise ValueError("incident_gradient must have shape (Q, ...batch, C, 2)")
    h_batch = h_values.shape[:-2]
    k, alpha, eta, field_gradient, _, _ = xp.broadcast_arrays(
        k,
        alpha,
        eta,
        field_gradient,
        incident_gradient[0, ..., 0, 0],
        xp.zeros(h_batch, dtype=point.dtype, device=point.device),
    )
    parameter = t[(...,) + (None,) * k.ndim]
    missing = k.ndim - (incident_gradient.ndim - 3)
    incident_gradient = incident_gradient[(slice(None),) + (None,) * missing + (...,)]
    dr_g = -xp.sum(incident_gradient * h_shape.x(parameter), axis=-1)
    direct = point_field_shape_derivative(
        density, point, shape=shape, h_shape=h_shape, k=k, alpha=alpha, eta=eta, n=n
    )
    return objective_derivative(
        k=k,
        shape=shape,
        alpha=alpha,
        eta=eta,
        n=n,
        phi=density,
        adjoint=adjoint,
        dr_j=xp.real(xp.conj(field_gradient) * direct),
        dr_g=dr_g,
        h_shape=h_shape,
    )
