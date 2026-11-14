"""Adjoint and central-difference sensitivity of a point-field objective."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import NystromInterpolant, Shape

from ._acoustic import near_field, scattering_dirichlet
from ._objective import point_objective_shape_derivative


@dataclass(frozen=True)
class SensitivityResult:
    """
    Scattered field, adjoint derivative, and central-difference derivatives.

    ``field`` is the complex scattered field, and ``adjoint`` is the real
    derivative of its squared distance from the target. ``central_field``
    and ``central_objective`` contain one approximation per requested step.
    """

    field: Array
    adjoint: Array
    central_field: Array
    central_objective: Array


@dataclass(frozen=True)
class _PerturbedShape:
    shape: Shape
    perturbation: Shape
    step: float

    def x(self, t: Array, /) -> Array:
        return self.shape.x(t) + self.step * self.perturbation.x(t)

    def dx(self, t: Array, /) -> Array:
        return self.shape.dx(t) + self.step * self.perturbation.dx(t)

    def ddx(self, t: Array, /) -> Array:
        return self.shape.ddx(t) + self.step * self.perturbation.ddx(t)


def adjoint_central_derivative(
    *,
    shape: Sequence[Shape],
    h_shape: Sequence[Shape],
    point: Array,
    target: Array,
    k: Array,
    alpha: Array,
    eta: Array,
    incident_field: Callable[[Array], Array],
    incident_field_grad: Callable[[Array], Array],
    n_solve: int,
    n_quad: int,
    steps: Sequence[float],
) -> SensitivityResult:
    r"""
    Compare an adjoint shape derivative with central differences.

    The objective is $|u_{\mathrm{scat}}(x_0)-c|^2$. Each boundary is
    perturbed as $x_j \pm \epsilon h_j$. Forward and adjoint solves use
    $2n_{\rm solve}-1$ nodes, while field and sensitivity integrations use
    $2n_{\rm quad}-1$ nodes. These parameters may differ. With different
    grids the continuous-adjoint quadrature is generally not exactly the
    derivative of the discrete Nyström solve; their difference converges away
    as both discretizations are resolved.

    Parameters
    ----------
    shape, h_shape : Sequence[Shape]
        Boundaries and their perturbations, each of (...) -> (..., 2).
    point : Array
        Observation point $x_0$ of shape (2,).
    target : Array
        Desired scattered field $c$ of shape ().
    k, alpha, eta : Array
        Scalar wave number and coupling parameters of shape ().
    incident_field : Callable[[Array], Array]
        Incident field of (..., 2) -> (...,).
    incident_field_grad : Callable[[Array], Array]
        Spatial incident-field gradient of (..., 2) -> (..., 2).
    n_solve, n_quad : int
        Nyström and integration parameters, respectively; both must be positive.
    steps : Sequence[float]
        Positive central-difference steps. An empty sequence skips them.

    Returns
    -------
    SensitivityResult
        Field and adjoint derivative of shape (), plus complex field and real
        objective central differences of shape (len(steps),).

    """
    check_shapes("D,D", point, (2,), names="point,coordinate_dimension")
    if point.shape != (2,) or any(value.ndim != 0 for value in (target, k, alpha, eta)):
        raise ValueError("point must have shape (2,) and wave parameters and target must be scalar")
    if not shape or len(shape) != len(h_shape):
        raise ValueError("shape and h_shape must have the same nonzero length")
    if n_solve < 1 or n_quad < 1 or any(not math.isfinite(step) or step <= 0 for step in steps):
        raise ValueError(
            "quadrature parameters and central-difference steps must be positive and finite"
        )
    xp = array_namespace(point, target, k, alpha, eta)

    def field_for(boundaries: Sequence[Shape], /) -> tuple[Array, NystromInterpolant]:
        density = scattering_dirichlet(
            k=k, shape=boundaries, incident_field=incident_field, alpha=alpha, eta=eta, n=n_solve
        )
        field = near_field(density, point, shape=boundaries, k=k, alpha=alpha, eta=eta, n=n_quad)
        return field, density

    field, density = field_for(shape)
    derivative = point_objective_shape_derivative(
        shape=shape,
        density=density,
        field=field,
        point=point,
        target=target,
        k=k,
        alpha=alpha,
        eta=eta,
        incident_field_grad=incident_field_grad,
        n_solve=n_solve,
        n_quad=n_quad,
    )(h_shape, range(len(shape)))

    central_field = []
    central_objective = []
    for step in steps:
        plus = [_PerturbedShape(s, h, step) for s, h in zip(shape, h_shape, strict=True)]
        minus = [_PerturbedShape(s, h, -step) for s, h in zip(shape, h_shape, strict=True)]
        field_plus, _ = field_for(plus)
        field_minus, _ = field_for(minus)
        central_field.append((field_plus - field_minus) / (2 * step))
        central_objective.append(
            (xp.abs(field_plus - target) ** 2 - xp.abs(field_minus - target) ** 2) / (2 * step)
        )
    empty_real = xp.empty(0, dtype=point.dtype, device=point.device)
    empty_complex = xp.empty(0, dtype=field.dtype, device=point.device)
    return SensitivityResult(
        field=field,
        adjoint=derivative,
        central_field=xp.stack(central_field) if central_field else empty_complex,
        central_objective=xp.stack(central_objective) if central_objective else empty_real,
    )
