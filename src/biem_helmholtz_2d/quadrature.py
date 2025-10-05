from array_api_compat import array_namespace
from array_api._2024_12 import Array, ArrayNamespaceFull

def trapezoidal_quadrature(n: int, /, *, xp: ArrayNamespaceFull) -> tuple[Array, Array]:
    r"""Trapezoidal quadrature.

    Returns $x_j$ and $w_j$, where

    $$
    \int_0^{2\pi} f(x) dx
    \approx \sum_{j=0}^{2n-1} w_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Number of discretization points / 2

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n,).
        and weights $w_j$ of shape (2n,).
    """
    x = xp.pi * xp.arange(2 * n) / n
    w = xp.full((1,), 2 * xp.pi / (2 * n))
    return x, w

def kussmaul_martensen_kress_quadrature(n: int, /, *, xp: ArrayNamespaceFull, x: Array | None = None) -> tuple[Array, Array]:
    r"""Kussmaul-Martensen (Kress) quadrature.

    Returns $y_j$ and $R_j (t)$, where

    $$
    \int_0^{2\pi} \log \left(4 \sin^2 \frac{x - y}{2}\right) f(y) dy
    \approx \sum_{j=0}^{2n-1} R_j (y) f(y_j)
    $$

    Parameters
    ----------
    n : int
        Number of discretization points / 2
    x : Array
        The target points of shape (...,).
        If None, it will be set to 0.

    Returns
    -------
    Array
        The roots $y_j$ of shape (2n,).
        and weights $T_j (x)$ of shape (..., 2n).
    """
    if x is None:
        t_ = xp.array(0.0)
    else:
        t_ = x
    x = xp.pi * xp.arange(2 * n) / n
    m = xp.arange(1, n)
    w = - 2 * xp.pi / n * xp.sum(
        1 / m * xp.cos(m * (t_[..., None, None] - x[:, None])), axis=-1
    ) - xp.pi / n**2 * xp.cos(n * (t_[..., None] - x))
    return x, w

def garrick_wittich_quadrature(n: int, /, *, xp: ArrayNamespaceFull, x: Array | None = None) -> tuple[Array, Array]:
    r"""Garrick-Wittich quadrature.

    Returns $y_j$ and $T_j (x)$, where

    $$
    \int_0^{2\pi} \cot \frac{x - y}{2} f(y) dy
    \approx \sum_{j=0}^{2n-1} T_j (x) f(y_j)
    $$

    Parameters
    ----------
    n : int
        Number of discretization points / 2
    x : Array
        The target points of shape (...,).
        If None, it will be set to 0.

    Returns
    -------
    Array
        The roots $y_j$ of shape (2n,).
        and weights $R_j (x)$ of shape (..., 2n).
    """
    if x is None:
        t_ = xp.asarray(0.0)
    else:
        t_ = x
    x = xp.pi * xp.arange(2 * n) / n
    m = xp.arange(1, n)
    w = - 1 / n * xp.sum(
        m * xp.cos(m * (t_[..., None, None] - x[:, None])), axis=-1
    ) - 1/ 2 * xp.cos(n * (t_[..., None] - x))
    return x, w