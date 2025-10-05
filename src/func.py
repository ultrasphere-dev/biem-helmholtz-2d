from array_api_compat import array_namespace
from array_api._2024_12 import Array, ArrayNamespaceFull

def kussmaul_martensen_kress_quadrature(n: int, /, *, xp: ArrayNamespaceFull, t: Array | None = None) -> tuple[Array, Array]:
    r"""Kussmaul-Martensen (Kress) quadrature.

    Returns $R_j (t)$$, where

    $$
    \int_0^{2\pi} \log \left(4 \sin^2 \frac{t - \tau}{2}\right) f(\tau) d\tau
    \approx \sum_{j=0}^{2n-1} R_j (t) f(t_j)
    $$

    Parameters
    ----------
    n : int
        Number of discretization points / 2
    t : Array
        The target points of shape (...,).
        If None, it will be set to 0.

    Returns
    -------
    Array
        The roots $t_k$ of shape (2n,).
        and weights $R_j (t)$ of shape (..., 2n).
    """
    if t is None:
        t_ = xp.array(0.0)
    else:
        t_ = t
    x = xp.pi * xp.arange(2 * n) / n
    m = xp.arange(1, n)
    w = - 2 * xp.pi / n * xp.sum(
        1 / m * xp.cos(m * (t_[..., None, None] - x[:, None])), axis=-1
    ) - xp.pi / n**2 * xp.cos(n * (t_[..., None] - x))
    return x, w

def garrick_wittich_quadrature(n: int, /, *, xp: ArrayNamespaceFull, t: Array | None = None) -> tuple[Array, Array]:
    r"""Garrick-Wittich quadrature.

    Returns $T_j (t)$$, where

    $$
    \int_0^{2\pi} \cot \frac{t - \tau}{2} f(\tau) d\tau
    \approx \sum_{j=0}^{2n-1} R_j (t) f(t_j)
    $$

    Parameters
    ----------
    n : int
        Number of discretization points / 2
    t : Array
        The target points of shape (...,).
        If None, it will be set to 0.

    Returns
    -------
    Array
        The roots $t_k$ of shape (2n,).
        and weights $R_j (t)$ of shape (..., 2n).
    """
    if t is None:
        t_ = xp.array(0.0)
    else:
        t_ = t
    x = xp.pi * xp.arange(2 * n) / n
    m = xp.arange(1, n)
    w = - 1 / n * xp.sum(
        m * xp.cos(m * (t_[..., None, None] - x[:, None])), axis=-1
    ) - 1/ 2 * xp.cos(n * (t_[..., None] - x))
    return x, w