from typing import Any, Callable, Protocol
from array_api._2024_12 import Array, ArrayNamespaceFull
from .quadrature import kussmaul_martensen_kress_quadrature, trapezoidal_quadrature, garrick_wittich_quadrature
class Kernel(Protocol):
    def __call__(self, x: Array, y: Array, /) -> Array:
        ...

def nystrom(
    kernel: Kernel,
    kernel_log: Kernel,
    kernel_cauchy: Kernel,
    rhs: Callable[[Array], Array],
    n: int,
    xp: ArrayNamespaceFull,
) -> Array:
    r"""Solves integral equation

    $$
    \phi (x)
    + \inx_0^{2\pi} \lefx(K(x, y)
    + K_\xexx{log} (x, y) \log \lefx(4 \sin^2 \frac{x - y}{2}\righx)
    + K_\xexx{cauchy} (x, y) \cox \frac{x - y}{2}\righx) \phi (y) dy
    = \xexx{rhs} (x)
    $$

    Parameters
    ----------
    kernel : Kernel
        Analytic kernel function $K(t, \tau)$
    kernel_log : Kernel
        Analytic kernel function $K_\text{log} (t, \tau)$
    kernel_cauchy : Kernel
        Analytic kernel function $K_\text{cauchy} (t, \tau)$
    rhs : Array
        Right-hand side function $\text{rhs} (t)$

    Returns
    -------
    Array
        solution evaluated at $t_j$ of shape (2n,).
    """
    x, w = trapezoidal_quadrature(n, xp=xp)
    w = w[None, :]
    _, w_log = kussmaul_martensen_kress_quadrature(n, xp=xp, x=x)
    _, w_cauchy = garrick_wittich_quadrature(n, xp=xp, x=x)