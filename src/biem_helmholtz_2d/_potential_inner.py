from __future__ import annotations

from typing import Literal

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from ._scipy_wrapper import scipy_hankel1


def dlp_kernel(
    point: Array,
    /,
    *,
    shape_x: Array,
    shape_dx: Array,
    k: Array,
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
    shape_x : Array
        Boundary positions $x(\tau)$ of shape (..., 2).
    shape_dx : Array
        Boundary derivatives $x'(\tau)$ of shape (..., 2).
    k : Array
        Wave number of shape (...,).

    Returns
    -------
    Array
        Kernel value $\widetilde{\mathcal D}(x_0, \tau)$ of shape (...,).

    """
    xp = array_namespace(point, k, shape_x, shape_dx)
    diff = point - shape_x
    r = xp.linalg.vector_norm(diff, axis=-1)
    h1 = scipy_hankel1(1, k * r)
    ny = xp.stack([shape_dx[..., 1], -shape_dx[..., 0]], axis=-1)
    ny_dot = xp.sum(ny * diff, axis=-1)
    return (1j * k / 4) * (ny_dot / r) * h1


def slp_kernel(
    point: Array,
    /,
    *,
    shape_x: Array,
    shape_dx: Array,
    k: Array,
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
    shape_x : Array
        Boundary positions $x(\tau)$ of shape (..., 2).
    shape_dx : Array
        Boundary derivatives $x'(\tau)$ of shape (..., 2).
    k : Array
        Wave number of shape (...,).

    Returns
    -------
    Array
        Kernel value $\widetilde{\mathcal S}(x_0, \tau)$ of shape (...,).

    """
    xp = array_namespace(point, k, shape_x, shape_dx)
    diff = point - shape_x
    r = xp.linalg.vector_norm(diff, axis=-1)
    h0 = scipy_hankel1(0, k * r)
    jac = xp.linalg.vector_norm(shape_dx, axis=-1)
    return (1j / 4) * h0 * jac


def kernel(
    point: Array,
    /,
    *,
    shape_x: Array,
    shape_dx: Array,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Array:
    r"""Near-field kernel of $\alpha\mathrm{DL}-i\eta\mathrm{SL}$."""
    return alpha * dlp_kernel(
        point, shape_x=shape_x, shape_dx=shape_dx, k=k
    ) - 1j * eta * slp_kernel(point, shape_x=shape_x, shape_dx=shape_dx, k=k)


def potential(
    x: Array,
    func: Array,
    /,
    *,
    kind: Literal["single", "double"],
    shape_x: Array,
    shape_dx: Array,
    k: Array,
) -> Array:
    r"""
    Evaluate a single- or double-layer potential at exterior points.

    Integrates the selected kernel against $\phi$ over the boundary.

    Parameters
    ----------
    x : Array
        Evaluation points of shape (*B, 2).
    func : Array
        Density values at equally spaced nodes over one $2\pi$ period of shape (..., Q).
    kind : {"single", "double"}
        Layer potential to evaluate.
    shape_x, shape_dx : Array
        Boundary positions and derivatives of shape (..., Q, 2).
    k : Array
        Wave number of shape (...,).

    Returns
    -------
    Array
        Layer potential of shape (...,).

    """
    xp = array_namespace(x, func, shape_x, shape_dx, k)
    kernel = {"single": slp_kernel, "double": dlp_kernel}[kind]
    values = kernel(x[..., None, :], shape_x=shape_x, shape_dx=shape_dx, k=k[..., None]) * func
    return xp.sum(values, axis=-1) * (2 * xp.pi / values.shape[-1])
