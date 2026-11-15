from __future__ import annotations

from array_api.latest import Array
from array_api_compat import array_namespace

from ._hankel import hankel_h1_h2
from ._is_close import is_close


def A1(
    *,
    t: Array,
    tau: Array,
    x_t: Array,
    x_tau: Array,
    dx_tau: Array,
    ddx_tau: Array,
    eps: float = 0,
) -> Array:
    r"""
    Kernel factor $A_1(t, \tau)$ for the double-layer potential.

    $$
    A_1(t, \tau) := \frac{n^*(\tau) \cdot (x(t) - x(\tau))}{|x(t) - x(\tau)|^2}
    $$

    Parameters
    ----------
    t : Array
        Target nodes $t$ of shape (...,).
    tau : Array
        Source nodes $\tau$ of shape (...,).
    x_t, x_tau : Array
        Boundary positions $x(t)$ and $x(\tau)$ of shape (..., 2).
    dx_tau, ddx_tau : Array
        Boundary derivatives $x'(\tau)$ and $x''(\tau)$ of shape (..., 2).
    eps : float
        If ``abs(tau - t) <= eps``, replace $A_1(t, \tau)$ by the diagonal limit value.

    Returns
    -------
    Array
        Values of $A_1(t, \tau)$ of shape (...,).

    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    xp = array_namespace(t, tau, x_t, x_tau, dx_tau, ddx_tau)
    diff = x_t - x_tau
    normal = xp.stack([dx_tau[..., 1], -dx_tau[..., 0]], axis=-1)
    result = xp.sum(normal * diff, axis=-1) / xp.sum(diff**2, axis=-1)
    limit = (ddx_tau[..., 0] * dx_tau[..., 1] - ddx_tau[..., 1] * dx_tau[..., 0]) / (
        2 * xp.sum(dx_tau**2, axis=-1)
    )
    return xp.where(is_close(t, tau, eps), limit, result)


def slp_kernel_split(
    *,
    t: Array,
    tau: Array,
    k: Array,
    x_t: Array,
    x_tau: Array,
    dx_tau: Array,
    eps: float = 0,
) -> tuple[Array, Array]:
    r"""
    Split single-layer kernel into log-singular and analytic parts.

    Parameters
    ----------
    t : Array
        Target nodes $t$ of shape (...,).
    tau : Array
        Source nodes $\tau$ of shape (...,).
    k : Array
        Wave number $k$ of shape (...,).
    x_t, x_tau : Array
        Boundary positions $x(t)$ and $x(\tau)$ of shape (..., 2).
    dx_tau : Array
        Boundary derivative $x'(\tau)$ of shape (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace the singular value of $S(t, \tau)$.

    Returns
    -------
    Array
        Log-singular coefficient of $S(t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $S(t, \tau)$ of shape (...,).

    """
    xp = array_namespace(t, tau, k, x_t, x_tau, dx_tau)
    jac = xp.linalg.vector_norm(dx_tau, axis=-1)
    argument = k * xp.linalg.vector_norm(x_t - x_tau, axis=-1)
    h1, h2 = hankel_h1_h2(tau, order=0, f=argument, fprime0=k * jac, eps=eps, t_singularity=t)
    return (1j / 4) * h1 * jac, (1j / 4) * h2 * jac


def dlp_kernel_split(
    *,
    t: Array,
    tau: Array,
    k: Array,
    x_t: Array,
    x_tau: Array,
    dx_tau: Array,
    ddx_tau: Array,
    eps: float = 0,
) -> tuple[Array, Array]:
    r"""
    Split double-layer kernel into log-singular and analytic parts.

    Parameters
    ----------
    t : Array
        Target nodes $t$ of shape (...,).
    tau : Array
        Source nodes $\tau$ of shape (...,).
    k : Array
        Wave number $k$ of shape (...,).
    x_t, x_tau : Array
        Boundary positions $x(t)$ and $x(\tau)$ of shape (..., 2).
    dx_tau, ddx_tau : Array
        Boundary derivatives $x'(\tau)$ and $x''(\tau)$ of shape (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace the singular value of $D(t, \tau)$.

    Returns
    -------
    Array
        Log-singular coefficient of $D(t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $D(t, \tau)$ of shape (...,).

    """
    xp = array_namespace(t, tau, k, x_t, x_tau, dx_tau, ddx_tau)
    argument = k * xp.linalg.vector_norm(x_t - x_tau, axis=-1)
    h1, h2 = hankel_h1_h2(tau, order=1, f=argument, fprime0=None, eps=eps, t_singularity=t)
    a1 = A1(t=t, tau=tau, x_t=x_t, x_tau=x_tau, dx_tau=dx_tau, ddx_tau=ddx_tau, eps=eps)
    return (1j / 4) * h1 * a1, (1j / 4) * h2 * a1
