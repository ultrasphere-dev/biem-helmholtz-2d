from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import trapezoidal_quadrature

from ._scipy_wrapper import scipy_hankel1


def slp(
    x: Array,
    func: Callable[[Array], Array],
    /,
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    k: Array,
    n: int,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> Array:
    r"""
    Single-layer potential kernel.

    Parameters
    ----------
    x : Array
        Evaluation points of shape (*B, 2).
    func : Callable[[Array], Array]
        Function to apply the single-layer potential to, of shape (..., *B) -> (..., *B).
    shape_x : Callable[[Array], Array]
        Boundary parametrization of shape (*B, 2).
    shape_dx : Callable[[Array], Array]
        First derivative of the boundary parametrization of shape (*B, 2).
    k : Array
        Wave number of shape (*B,).
    n : int
        The maximum order - 1.
    t_start : float | None
        Grid shift $t_\mathrm{start}$, with $x_j := t_\mathrm{start} + 2\pi j / (2n-1)$.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$. Mutually exclusive with
        ``t_start``.

    Returns
    -------
    Array
        Values of the single-layer potential kernel of shape (...,).

    """
    xp = array_namespace(x, k)
    dtype = xp.result_type(x, k)
    device = x.device
    t, w = trapezoidal_quadrature(
        n, t_start=t_start, t_start_factor=t_start_factor, xp=xp, dtype=dtype, device=device
    )
    diff = x[..., None, :] - shape_x(t)
    dist = xp.linalg.vector_norm(diff, axis=-1)
    h1 = scipy_hankel1(0, k[..., None] * dist)
    jac = xp.linalg.vector_norm(shape_dx(t), axis=-1)
    return (1j / 4) * xp.sum(h1 * jac * w * func(t), axis=-1)


def dlp(
    x: Array,
    func: Callable[[Array], Array],
    /,
    shape_x: Callable[[Array], Array],
    shape_dx: Callable[[Array], Array],
    k: Array,
    n: int,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> Array:
    r"""
    Double-layer potential kernel.

    Parameters
    ----------
    x : Array
        Evaluation points of shape (*B, 2).
    func : Callable[[Array], Array]
        Function to apply the double-layer potential to, of shape (..., *B) -> (..., *B).
    shape_x : Callable[[Array], Array]
        Boundary parametrization of shape (*B, 2).
    shape_dx : Callable[[Array], Array]
        First derivative of the boundary parametrization of shape (*B, 2).
    k : Array
        Wave number of shape (*B,).
    n : int
        The maximum order - 1.
    t_start : float | None
        Grid shift $t_\mathrm{start}$, with $x_j := t_\mathrm{start} + 2\pi j / (2n-1)$.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$. Mutually exclusive with
        ``t_start``.

    Returns
    -------
    Array
        Values of the double-layer potential kernel of shape (...,).

    """
    xp = array_namespace(x, k)
    dtype = xp.result_type(x, k)
    device = x.device
    t, w = trapezoidal_quadrature(
        n, t_start=t_start, t_start_factor=t_start_factor, xp=xp, dtype=dtype, device=device
    )
    diff = x[..., None, :] - shape_x(t)
    dist = xp.linalg.vector_norm(diff, axis=-1)
    h1 = scipy_hankel1(1, k[..., None] * dist)
    ny = xp.stack([shape_dx(t)[..., 1], -shape_dx(t)[..., 0]], axis=-1)
    integrand = xp.sum(ny * diff, axis=-1) / dist * h1
    return (1j * k / 4) * xp.sum(integrand * w * func(t), axis=-1)
