from typing import Any, Callable, Protocol
from array_api._2024_12 import Array, ArrayNamespaceFull
from .quadrature import kussmaul_martensen_kress_quadrature, trapezoidal_quadrature, garrick_wittich_quadrature
import attrs

class KernelResult(Protocol):
    analytic: Array
    singular_log: Array
    singular_cauchy: Array

@attrs.frozen(kw_only=True)
class KernelResultImpl(KernelResult):
    analytic: Array
    singular_log: Array
    singular_cauchy: Array

class Kernel(Protocol):
    def __call__(self, x: Array, y: Array, /) -> KernelResult:
        """Kernel function.

        Parameters
        ----------
        x : Array
            An array of shape (...,).
        y : Array
            An array of shape (...,).

        Returns
        -------
        KernelResult
            The kernel function values of shape (..., ...(fixed)).
        """

def nystrom_lhs(
    kernel: Kernel,
    n: int,
    xp: ArrayNamespaceFull,
) -> tuple[Array, Array]:
    r"""Returns the left-hand side matrix $A$ of the Nystrom method for the integral equation

    $$
    \phi (x)
    + \int_0^{2\pi} \left(K(x, y)
    + K_\text{log} (x, y) \log \left(4 \sin^2 \frac{x - y}{2}\right)
    + K_\text{cauchy} (x, y) \cot \frac{x - y}{2}\right) \phi (y) dy
    = \text{rhs} (x)
    $$

    Parameters
    ----------
    kernel : Kernel
        Kernel function $K (x, y)$, $K_\text{log} (x, y)$, and $K_\text{cauchy} (x, y)$
    n : int
        Number of discretization points / 2
    xp : ArrayNamespaceFull
        The array namespace.

    Returns
    -------
    tuple[Array, Array]
        The roots $x_j$ of shape (2n,)
        and the left-hand side matrix $A$ of shape (2n, 2n).
    """
    x, w = trapezoidal_quadrature(n, xp=xp)
    y = x[None, :]
    w = w[None, :]
    _, w_log = kussmaul_martensen_kress_quadrature(n, xp=xp, x=x)
    _, w_cauchy = garrick_wittich_quadrature(n, xp=xp, x=x)
    x = x[:, None]
    # x: (2n, 1), y: (1, 2n), w: (1, 2n), w_log: (2n, 2n), w_cauchy: (2n, 2n)
    k = kernel(x, y)
    A = xp.eye(2 * n, dtype=x.dtype, device=x.device) + (
        k.analytic * w
        + k.singular_log * w_log
        + k.singular_cauchy * w_cauchy
    )
    return x[:, 0], A