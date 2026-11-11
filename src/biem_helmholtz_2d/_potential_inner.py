from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from ._quadrature import trapezoidal_quadrature
from ._scipy_wrapper import scipy_hankel1


def dlp_kernel(
    point: Array,
    /,
    *,
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    k: Array,
    tau: Array,
) -> Array:
    r"""
    Kernel of the double-layer potential at an exterior point.

    $$
    \widetilde{\mathcal D}(x_0, \tau)
    = \frac{\mathrm i k}{4}\,
      H_1^{(1)}(k|x_0 - x(\tau)|)\,
      \frac{n^*(\tau) \cdot (x_0 - x(\tau))}{|x_0 - x(\tau)|}
    $$

    Parameters
    ----------
    point : Array
        Exterior evaluation point $x_0$ of shape (..., 2).
    shape_x : Callable[[Array], Array]
        Boundary parametrisation $x(\tau)$ of (...,) -> (..., 2).
    shape_dx : Callable[[Array], Array]
        Derivative $x'(\tau)$ of (...,) -> (..., 2).
    k : Array
        Wave number of shape (...,).
    tau : Array
        Boundary parameter $\tau$ of shape (...,).

    Returns
    -------
    Array
        Kernel value $\widetilde{\mathcal D}(x_0, \tau)$ of shape (...,).

    """
    xp = array_namespace(point, k, tau)
    diff = point - shape_x(tau)
    r = xp.linalg.vector_norm(diff, axis=-1)
    h1 = scipy_hankel1(1, k * r)
    ny = xp.stack([shape_dx(tau)[..., 1], -shape_dx(tau)[..., 0]], axis=-1)
    ny_dot = xp.sum(ny * diff, axis=-1)
    return (1j * k / 4) * (ny_dot / r) * h1


def slp_kernel(
    point: Array,
    /,
    *,
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    k: Array,
    tau: Array,
) -> Array:
    r"""
    Kernel of the single-layer potential at an exterior point.

    $$
    \widetilde{\mathcal S}(x_0, \tau)
    = \frac{\mathrm i}{4}\,
      H_0^{(1)}(k|x_0 - x(\tau)|)\,
      |x'(\tau)|
    $$

    Parameters
    ----------
    point : Array
        Exterior evaluation point $x_0$ of shape (..., 2).
    shape_x : Callable[[Array], Array]
        Boundary parametrisation $x(\tau)$ of (...,) -> (..., 2).
    shape_dx : Callable[[Array], Array]
        Derivative $x'(\tau)$ of (...,) -> (..., 2).
    k : Array
        Wave number of shape (...,).
    tau : Array
        Boundary parameter $\tau$ of shape (...,).

    Returns
    -------
    Array
        Kernel value $\widetilde{\mathcal S}(x_0, \tau)$ of shape (...,).

    """
    xp = array_namespace(point, k, tau)
    diff = point - shape_x(tau)
    r = xp.linalg.vector_norm(diff, axis=-1)
    h0 = scipy_hankel1(0, k * r)
    jac = xp.linalg.vector_norm(shape_dx(tau), axis=-1)
    return (1j / 4) * h0 * jac


def potential(
    x: Array,
    func: Callable[[Array], Array],
    /,
    *,
    kind: Literal["single", "double"],
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    k: Array,
    n: int,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> Array:
    r"""
    Evaluate a single- or double-layer potential at exterior points.

    Integrates the selected kernel against $\phi$ over the boundary.

    Parameters
    ----------
    x : Array
        Evaluation points of shape (*B, 2).
    func : Callable[[Array], Array]
        Density $\phi$ of (..., *B) -> (..., *B).
    kind : {"single", "double"}
        Layer potential to evaluate.
    shape_x, shape_dx : Callable[[Array], Array]
        Boundary parametrization and derivative of (...,) -> (..., 2).
    k : Array
        Wave number of shape (*B,).
    n : int
        Quadrature order; the number of nodes is $2n-1$.
    t_start, t_start_factor : float or None
        Optional grid shift and grid shift as a multiple of the node spacing.

    Returns
    -------
    Array
        Layer potential of shape (...,).

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
    kernel = {"single": slp_kernel, "double": dlp_kernel}[kind]
    return xp.sum(
        kernel(x[..., None, :], shape_x=shape_x, shape_dx=shape_dx, k=k, tau=t) * w * func(t),
        axis=-1,
    )
