from enum import StrEnum
from typing import Any, Protocol

from array_api._2024_12 import Array, ArrayNamespaceFull

from .quadrature import (
    cot_power_shifted_quadrature,
    log_cot_power_shifted_quadrature,
    trapezoidal_quadrature,
)


class QuadratureType(StrEnum):
    NO_SINGULARITY = "no_singularity"
    LOG_COT_POWER = "log_cot_power"
    COT_POWER = "cot_power"


class KernelFunction(Protocol):
    def __call__(self, x: Array, y: Array, /) -> Array:
        """
        Kernel function.

        Parameters
        ----------
        x : Array
            An array of shape (...,).
        y : Array
            An array of shape (...,).

        Returns
        -------
        Array
            The kernel function values of shape (..., ...(fixed)).

        """
        ...


Kernel = dict[tuple[QuadratureType, int], KernelFunction]


def nystrom_lhs(
    a: KernelFunction,
    kernel: Kernel,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Returns the left-hand side matrix $A$ of the Nystrom method for the integral equation

    $$
    a(x) \phi (x)
    + \int_0^{2\pi} \Bigg( K_{\mathrm{reg}}(x, y)
    + \sum_{n\ge 0} K_{\log,n}(x, y)\,\log\left(4\sin^2\frac{x - y}{2}\right)\cot^n\!\left(\frac{x - y}{2}\right)
    + \sum_{n\ge 0} K_{\cot,n}(x, y)\,\cot^n\!\left(\frac{x - y}{2}\right) \Bigg)\,\phi (y)\,dy
    = \text{rhs} (x)
    $$

    Parameters
    ----------
    a : KernelFunction
        Multiplicative term $a(x)$.
    kernel : Kernel
        Kernel functions keyed by ``(QuadratureType, order)``.
    n : int
        Number of discretization points / 2
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    tuple[Array, Array]
        The roots $x_j$ of shape (2n - 1,)
        and the left-hand side matrix $A$ of shape (2n - 1, 2n - 1).

    """
    x, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    n_quad = 2 * n - 1
    y = x[None, :]

    w_scalar = w[0]
    idx = (
        xp.arange(n_quad, device=device, dtype=xp.int64)[:, None]
        + xp.arange(n_quad, device=device, dtype=xp.int64)[None, :]
    ) % n_quad

    x = x[:, None]
    weight_by_key: dict[tuple[QuadratureType, int], Array] = {}
    for quad_type, order in kernel:
        if quad_type == QuadratureType.NO_SINGULARITY:
            weight_by_key[(quad_type, order)] = w_scalar
        elif quad_type == QuadratureType.LOG_COT_POWER:
            _, w_log_vec = log_cot_power_shifted_quadrature(
                n, order, xp=xp, device=device, dtype=dtype
            )
            weight_by_key[(quad_type, order)] = xp.take(w_log_vec, idx)
        elif quad_type == QuadratureType.COT_POWER:
            _, w_cauchy_vec = cot_power_shifted_quadrature(
                n, order, xp=xp, device=device, dtype=dtype
            )
            weight_by_key[(quad_type, order)] = xp.take(w_cauchy_vec, idx)
        else:
            msg = f"Unsupported quadrature type: {quad_type}"
            raise ValueError(msg)

    terms = [
        kernel_fn(x, y) * weight_by_key[(quad_type, order)]
        for (quad_type, order), kernel_fn in kernel.items()
    ]
    a_vals = a(x[:, 0], x[:, 0])
    A = xp.eye(n_quad, dtype=dtype, device=device) * a_vals[:, None] + xp.sum(
        xp.stack(terms), axis=0
    )
    return x[:, 0], A
