from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull


def trapezoidal_quadrature(
    n: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Trapezoidal quadrature for [0, 2π].

    Returns $x_j$ and $w_j$, where

    $$
    \int_0^{2\pi} f(x) dx
    \approx \sum_{j=0}^{2n-1} w_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    xp: ArrayNamespaceFull
        The array namespace.
    device: Any
        The device.
    dtype: Any
        The dtype.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $w_j$ of shape (2n - 1,).

    """
    n_quad = 2 * n - 1
    two_pi = 2 * xp.pi
    j = xp.asarray(xp.arange(n_quad), dtype=dtype, device=device)
    x = two_pi * j / n_quad
    w = xp.full((1,), two_pi / n_quad, dtype=dtype, device=device)
    return x, w


def kussmaul_martensen_kress_quadrature(
    n: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Kussmaul-Martensen (Kress) quadrature.

    Returns $x_j$ and $R_j$, where

    Let $n' := 2n - 1$ and $x_j := 2\pi j / n'$.

    $$
    \int_0^{2\pi} \log \left(4 \sin^2 \frac{t}{2}\right) f(t) dt
    \approx \sum_{j=0}^{n'-1} R_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    xp: ArrayNamespaceFull
        The array namespace.
    device: Any
        The device.
    dtype: Any
        The dtype.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $R_j$ of shape (2n - 1,).

    """
    n_quad = 2 * n - 1
    two_pi = xp.asarray(xp.pi, dtype=dtype, device=device) + xp.asarray(
        xp.pi, dtype=dtype, device=device
    )
    j = xp.arange(n_quad)
    x = two_pi * j / n_quad
    m = xp.arange(1, n)
    w = (
        -(two_pi + two_pi)
        / n_quad
        * xp.sum(xp.cos(m[:, None] * x[None, :]) / m[:, None], axis=0)
    )
    return x, w


def garrick_wittich_quadrature(
    n: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Garrick-Wittich quadrature.

    Returns $x_j$ and $T_j$, where

    Let $n' := 2n - 1$ and $x_j := 2\pi j / n'$.

    $$
    p.v. \int_0^{2\pi} \cot \frac{t}{2} f'(t) dt
    \approx \sum_{j=0}^{n'-1} T_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    xp: ArrayNamespaceFull
        The array namespace.
    device: Any
        The device.
    dtype: Any
        The dtype.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $T_j$ of shape (2n - 1,).

    """
    n_quad = 2 * n - 1
    two_pi = xp.asarray(xp.pi, dtype=dtype, device=device) + xp.asarray(
        xp.pi, dtype=dtype, device=device
    )
    j = xp.arange(n_quad)
    x = two_pi * j / n_quad
    m = xp.arange(1, n)
    w = (
        -(two_pi + two_pi)
        / n_quad
        * xp.sum(m[:, None] * xp.cos(m[:, None] * x[None, :]), axis=0)
    )
    return x, w
