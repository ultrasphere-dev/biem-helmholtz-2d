from __future__ import annotations

from collections.abc import Callable, Sequence

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import NystromInterpolant, Shape, trapezoidal_quadrature

from ._acoustic import _density_component
from ._adjoint import objective_derivative, solve_adjoint
from ._potential_inner import dlp_kernel, slp_kernel
from ._potential_inner_derivative import potential_shape_derivative


def abs2_scattered_field(
    target: Array,
    /,
    *,
    field_value: Array,
) -> Array:
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


def grad_phi_abs2_scattered_field(
    point: Array,
    field_value: Array,
    /,
    *,
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
    target: Array,
) -> Callable[[Array], Array]:
    r"""
    Riesz representation of $d_\phi j$ under the $L^2$ sesquilinear form.

    For the objective $j(\phi) = |u(x_0) - c|^2$ with
    $u = (\alpha\mathcal D - i\eta\mathcal S)\phi$, the Frechet derivative
    with respect to $\phi$ is

    $$
    D_\phi j[\nu] = 2\operatorname{Re}\bigl(
        (\overline{u(x_0)} - \overline{c})\,(\alpha\mathcal D - i\eta\mathcal S)\nu(x_0)
    \bigr).
    $$

    Under the $L^2$ sesquilinear form $\langle f,g\rangle = \int f\,\overline g$,
    the Riesz representation is

    $$
    \operatorname{grad}_\phi j(\tau)
    = 2\,(u(x_0) - c)\,
      \overline{\bigl(\alpha\widetilde{\mathcal D}(x_0,\tau)
                    - i\eta\,\widetilde{\mathcal S}(x_0,\tau)\bigr)},
    $$

    where

    $$
    \widetilde{\mathcal S}(x_0,\tau) := G(x_0, x(\tau))\,|x'(\tau)|,
    \qquad
    \widetilde{\mathcal D}(x_0,\tau) := n(\tau)\cdot\nabla_y G(x_0, x(\tau))\,|x'(\tau)|
    $$

    are the kernels of $\mathrm{SL}_x$, $\mathrm{DL}_x$
    with jacobian multiplied, evaluated at $x_0$.
    The conjugate involves Hankel functions of the second kind
    $H_n^{(2)} = \overline{H_n^{(1)}}$.

    Parameters
    ----------
    point : Array
        Evaluation point $x_0$ of shape (..., 2).
    field_value : Array
        Scattered field $u(x_0)$ of shape (...,).
    shape : Shape
        Boundary parametrisation of (...,) -> (..., 2).
    k : Array
        Wave number of shape (...,).
    alpha : Array
        Coupling parameter $\alpha$ of shape (...,).
    eta : Array
        Coupling parameter $\eta$ of shape (...,).
    target : Array
        Target value $c$ of shape (...,).

    Returns
    -------
    Callable[[Array], Array]
        Riesz representation of $d_\phi j$ that takes boundary parameter
        $t$ of shape ``(...,)`` and returns values of shape ``(...,)``.

    """
    xp = array_namespace(point, field_value, k, alpha, eta, target)
    diff = field_value - target

    def grad(t_in: Array) -> Array:
        dlp = dlp_kernel(point, shape_x=shape.x, shape_dx=shape.dx, k=k, tau=t_in)
        slp = slp_kernel(point, shape_x=shape.x, shape_dx=shape.dx, k=k, tau=t_in)
        return 2 * diff * xp.conj(alpha * dlp - 1j * eta * slp)

    return grad


def point_objective_shape_derivative(
    *,
    shape: Sequence[Shape],
    density: NystromInterpolant,
    field: Array,
    point: Array,
    target: Array,
    k: Array,
    alpha: Array,
    eta: Array,
    incident_field_grad: Callable[[Array], Array],
    n_solve: int,
    n_quad: int,
) -> Callable[[Sequence[Shape], Sequence[int]], Array]:
    """
    Prepare the shape derivative of the point-field objective for reuse.

    The returned function takes boundary perturbations and a nonempty sequence
    of active boundary indices, which must include every nonzero perturbation.
    """
    xp = array_namespace(point, target, k, alpha, eta)
    t, _ = trapezoidal_quadrature(n_quad, xp=xp, device=point.device, dtype=point.dtype)
    gradients = [
        grad_phi_abs2_scattered_field(
            point, field, shape=boundary, k=k, alpha=alpha, eta=eta, target=target
        )
        for boundary in shape
    ]

    def grad_phi(parameter: Array, /) -> Array:
        return xp.stack([gradient(parameter) for gradient in gradients], axis=-1)

    adjoint = solve_adjoint(k=k, shape=shape, alpha=alpha, eta=eta, n=n_solve, grad_phi_j=grad_phi)
    incident_grads = [incident_field_grad(boundary.x(t)) for boundary in shape]

    def derivative(h_shape: Sequence[Shape], active: Sequence[int], /) -> Array:
        dr_g = xp.stack(
            [
                -xp.sum(grad * perturbation.x(t), axis=-1)
                for grad, perturbation in zip(incident_grads, h_shape, strict=True)
            ],
            axis=-1,
        )
        direct = []
        for index in active:
            boundary, perturbation = shape[index], h_shape[index]
            component = _density_component(density, index)
            ds, dd = (
                potential_shape_derivative(
                    point,
                    component,
                    kind=kind,
                    shape_x=boundary.x,
                    shape_dx=boundary.dx,
                    h=perturbation.x,
                    dh=perturbation.dx,
                    k=k,
                    n=n_quad,
                )
                for kind in ("single", "double")
            )
            direct.append(alpha * dd - 1j * eta * ds)
        dr_j = 2 * xp.real(xp.conj(field - target) * xp.sum(xp.stack(direct)))
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

    return derivative
