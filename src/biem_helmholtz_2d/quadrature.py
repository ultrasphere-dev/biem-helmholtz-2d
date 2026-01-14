from array_api._2024_12 import Array, ArrayNamespaceFull


def trapezoidal_quadrature(n: int, /, *, xp: ArrayNamespaceFull) -> tuple[Array, Array]:
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

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $w_j$ of shape (2n - 1,).

    """
    n_quad = 2 * n - 1
    x = 2 * xp.pi * xp.arange(n_quad) / n_quad
    w = xp.full((1,), 2 * xp.pi / n_quad)
    return x, w


def kussmaul_martensen_kress_quadrature(
    n: int, /, *, xp: ArrayNamespaceFull
) -> tuple[Array, Array]:
    r"""
    Kussmaul-Martensen (Kress) quadrature.

    Returns $x_j$ and $R_j$, where

    $$
    \int_0^{2\pi} \log \left(4 \sin^2 \frac{x}{2}\right) f(x) dx
    \approx \sum_{j=0}^{2n-1} R_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    xp: ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $R_j$ of shape (2n - 1,).

    """
    x = xp.pi * xp.arange(2 * n) / n
    m = xp.arange(1, n)
    w = -2 * xp.pi / n * xp.sum(
        1 / m * xp.cos(m * (t_[..., None, None] - x[:, None])), axis=-1
    ) - xp.pi / n**2 * xp.cos(n * (t_[..., None] - x))
    return x, w


def garrick_wittich_quadrature(
    n: int, /, *, xp: ArrayNamespaceFull
) -> tuple[Array, Array]:
    r"""
    Garrick-Wittich quadrature.

    Returns $x_j$ and $T_j (x)$, where

    $$
    p.v. \int_0^{2\pi} \cot \frac{x}{2} f(x) dy
    \approx \sum_{j=0}^{2n-1} T_j (x) f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    xp: ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $T_j$ of shape (2n - 1,).

    """
    if x is None:
        t_ = xp.asarray(0.0)
    else:
        t_ = x
    x = xp.pi * xp.arange(2 * n) / n
    m = xp.arange(1, n)
    w = -1 / n * xp.sum(
        m * xp.cos(m * (t_[..., None, None] - x[:, None])), axis=-1
    ) - 1 / 2 * xp.cos(n * (t_[..., None] - x))
    return x, w
