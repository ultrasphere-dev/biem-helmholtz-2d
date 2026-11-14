from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import trapezoidal_quadrature

from ._scipy_wrapper import scipy_hankel1


def slp_shape_derivative_kernel(
    point: Array,
    /,
    *,
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    h: Callable[[Array], Array],
    dh: Callable[[Array], Array],
    k: Array,
    tau: Array,
) -> Array:
    r"""
    Shape derivative of the single-layer potential kernel at an exterior point.

    For the kernel
    $\widetilde{\mathcal S}(x_0, \tau)
    = \frac{\mathrm i}{4} H_0^{(1)}(k r)\,|x'(\tau)|$
    with $r = |x_0 - x(\tau)|$, the shape derivative in direction $h$ is

    $$
    D_r[\widetilde{\mathcal S}][h]
    = \frac{\mathrm i}{4} \left[
        k\,H_1^{(1)}(k r)\,\frac{(x_0 - x(\tau)) \cdot h(\tau)}{r}\,|x'(\tau)|
        + H_0^{(1)}(k r)\,\frac{x'(\tau) \cdot h'(\tau)}{|x'(\tau)|}
    \right]
    $$

    Parameters
    ----------
    point : Array
        Exterior evaluation point $x_0$ of shape (..., 2).
    shape_x : Callable[[Array], Array]
        Boundary parametrisation $x(\tau)$ of (...,) -> (..., 2).
    shape_dx : Callable[[Array], Array]
        Derivative $x'(\tau)$ of (...,) -> (..., 2).
    h : Callable[[Array], Array]
        Shape perturbation $h(\tau)$ of (...,) -> (..., 2).
    dh : Callable[[Array], Array]
        Derivative $h'(\tau)$ of (...,) -> (..., 2).
    k : Array
        Wave number of shape (...,).
    tau : Array
        Boundary parameter $\tau$ of shape (...,).

    Returns
    -------
    Array
        Shape derivative kernel value of shape (...,).

    """
    xp = array_namespace(point, k, tau)
    diff = point - shape_x(tau)
    r = xp.linalg.vector_norm(diff, axis=-1)
    h_tau = h(tau)
    dh_tau = dh(tau)
    dx_tau = shape_dx(tau)
    jac = xp.linalg.vector_norm(dx_tau, axis=-1)

    diff_dot_h = xp.sum(diff * h_tau, axis=-1)
    dx_dot_dh = xp.sum(dx_tau * dh_tau, axis=-1)

    h0 = scipy_hankel1(0, k * r)
    h1 = scipy_hankel1(1, k * r)

    return (1j / 4) * (k * h1 * diff_dot_h / r * jac + h0 * dx_dot_dh / jac)


def dlp_shape_derivative_kernel(
    point: Array,
    /,
    *,
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    h: Callable[[Array], Array],
    dh: Callable[[Array], Array],
    k: Array,
    tau: Array,
) -> Array:
    r"""
    Shape derivative of the double-layer potential kernel at an exterior point.

    For the kernel
    $\widetilde{\mathcal D}(x_0, \tau)
    = \frac{\mathrm i k}{4}\,\frac{H_1^{(1)}(k r)}{r}\,n^*(\tau) \cdot (x_0 - x(\tau))$
    with $r = |x_0 - x(\tau)|$ and $n^*(\tau) = (x'_2(\tau), -x'_1(\tau))$,
    the shape derivative in direction $h$ is

    $$
    D_r[\widetilde{\mathcal D}][h]
    = \frac{\mathrm i k}{4} \left[
        -\left(\frac{k\,H_0^{(1)}(k r)}{r}
        - \frac{2\,H_1^{(1)}(k r)}{r^2}\right)
        \frac{(x_0 - x(\tau)) \cdot h(\tau)}{r}\,n^*(\tau)
        \cdot (x_0 - x(\tau))
    \right.
    $$
    $$
    \left.
        + \frac{H_1^{(1)}(k r)}{r}\,\bigl(n^{*\prime}[h](\tau)
        \cdot (x_0 - x(\tau)) - n^*(\tau) \cdot h(\tau)\bigr)
    \right]
    $$

    where $n^{*\prime}[h](\tau) = (h'_2(\tau), -h'_1(\tau))$.

    Parameters
    ----------
    point : Array
        Exterior evaluation point $x_0$ of shape (..., 2).
    shape_x : Callable[[Array], Array]
        Boundary parametrisation $x(\tau)$ of (...,) -> (..., 2).
    shape_dx : Callable[[Array], Array]
        Derivative $x'(\tau)$ of (...,) -> (..., 2).
    h : Callable[[Array], Array]
        Shape perturbation $h(\tau)$ of (...,) -> (..., 2).
    dh : Callable[[Array], Array]
        Derivative $h'(\tau)$ of (...,) -> (..., 2).
    k : Array
        Wave number of shape (...,).
    tau : Array
        Boundary parameter $\tau$ of shape (...,).

    Returns
    -------
    Array
        Shape derivative kernel value of shape (...,).

    """
    xp = array_namespace(point, k, tau)
    diff = point - shape_x(tau)
    r = xp.linalg.vector_norm(diff, axis=-1)
    h_tau = h(tau)
    dh_tau = dh(tau)
    dx_tau = shape_dx(tau)

    diff_dot_h = xp.sum(diff * h_tau, axis=-1)

    n_star = xp.stack([dx_tau[..., 1], -dx_tau[..., 0]], axis=-1)
    n_star_dot_diff = xp.sum(n_star * diff, axis=-1)
    n_star_dot_h = xp.sum(n_star * h_tau, axis=-1)

    n_star_prime = xp.stack([dh_tau[..., 1], -dh_tau[..., 0]], axis=-1)
    n_star_prime_dot_diff = xp.sum(n_star_prime * diff, axis=-1)

    h0 = scipy_hankel1(0, k * r)
    h1 = scipy_hankel1(1, k * r)

    # f1 = H_1(kr)/r, f1' = k H_0(kr)/r - 2 H_1(kr)/r^2
    f1 = h1 / r
    f1_prime = k * h0 / r - 2 * h1 / r**2

    return (1j * k / 4) * (
        -f1_prime * diff_dot_h / r * n_star_dot_diff + f1 * (n_star_prime_dot_diff - n_star_dot_h)
    )


def potential_shape_derivative(
    x: Array,
    func: Callable[[Array], Array],
    /,
    *,
    kind: Literal["single", "double"],
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    h: Callable[[Array], Array],
    dh: Callable[[Array], Array],
    k: Array,
    n: int,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> Array:
    r"""
    Shape derivative of a single- or double-layer potential at exterior points.

    Integrates the shape derivative of the selected kernel against $\phi$.

    Parameters
    ----------
    x : Array
        Exterior evaluation point $x_0$ of shape (*B, 2).
    func : Callable[[Array], Array]
        Density $\phi$ of (..., *B) -> (..., *B).
    kind : {"single", "double"}
        Layer potential to differentiate.
    shape_x, shape_dx : Callable[[Array], Array]
        Boundary parametrization and derivative of (...,) -> (..., 2).
    h, dh : Callable[[Array], Array]
        Shape perturbation and derivative of (...,) -> (..., 2).
    k : Array
        Wave number of shape (*B,).
    n : int
        Quadrature parameter; the number of nodes is $2n-1$.
    t_start, t_start_factor : float or None
        Optional grid shift and grid shift as a multiple of the node spacing.

    Returns
    -------
    Array
        Shape derivative of the layer potential of shape (...,).

    """
    xp = array_namespace(x, k)
    t, w = trapezoidal_quadrature(
        n,
        t_start=t_start,
        t_start_factor=t_start_factor,
        xp=xp,
        dtype=xp.result_type(x, xp.real(k)),
        device=x.device,
    )
    kernel = {"single": slp_shape_derivative_kernel, "double": dlp_shape_derivative_kernel}[kind]
    return xp.sum(
        kernel(x[..., None, :], shape_x=shape_x, shape_dx=shape_dx, h=h, dh=dh, k=k, tau=t)
        * w
        * func(t),
        axis=-1,
    )
