from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from biem_helmholtz_2d._hankel import hankel_h1_h2

from ._is_close import is_close


def D_t(
    t: Array,
    tau: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    ddx: Callable[[Array], Array],
    /,
    *,
    eps: float = 0.0,
) -> Array:
    r"""
    Kernel factor for the double-layer potential.

    Parameters
    ----------
    t : Array
        Source nodes of shape (...,).
    tau : Array
        Target nodes of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization returning shape (..., 2).
    dx : Callable[[Array], Array]
        First derivative of the parametrization of shape (..., 2).
    ddx : Callable[[Array], Array]
        Second derivative of the parametrization of shape (..., 2).
    eps : float
        If ``abs(tau - t) <= eps``, replace by the diagonal limit value.

    Returns
    -------
    Array
        Values of $D_t$ of shape (...,).

    """
    if eps < 0:
        msg = "eps must be non-negative."
        raise ValueError(msg)

    xp = array_namespace(t, tau)
    x_t = x(t)
    x_tau = x(tau)
    dx_tau = dx(tau)
    ddx_tau = ddx(tau)

    # non-limit part
    diff = x_t - x_tau
    dist = xp.linalg.vector_norm(diff, axis=-1)
    outward_unnormalized = xp.stack([dx_tau[..., 1], -dx_tau[..., 0]], axis=-1)
    result = xp.sum(outward_unnormalized * diff, axis=-1) / (dist**2)

    # limit part
    near0 = is_close(t, tau, eps)
    limit = (dx_tau[..., 0] * ddx_tau[..., 1] - dx_tau[..., 1] * ddx_tau[..., 0]) / (
        -2 * xp.linalg.vector_norm(dx_tau, axis=-1) ** 2
    )
    return xp.where(near0, limit, result)


def slp_kernel_split(
    t: Array,
    tau: Array,
    k: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    /,
    *,
    eps: float = 0.0,
) -> tuple[Array, Array]:
    r"""
    Split single-layer kernel into log-singular and analytic parts.

    Parameters
    ----------
    t : Array
        Source nodes of shape (...,).
    tau : float
        Target node of shape (...,).
    k : float
        Wave number of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization returning shape (..., 2).
    dx : Callable[[Array], Array]
        First derivative of the parametrization of shape (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace by the diagonal limit value.

    Returns
    -------
    Array
        Log-singular coefficient of shape (N',).
    Array
        Analytic remainder of shape (N',).

    """
    xp = array_namespace(t, tau)
    x_t = x(t)
    dx_tau = dx(tau)
    jac_tau = xp.sqrt(xp.sum(dx_tau**2, axis=-1))

    def fval(t_in: Array) -> Array:
        diff = x_t - x(t_in)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        return k * r

    h1, h2 = hankel_h1_h2(
        tau,
        order=0,
        f=fval,
        fprime0=k * jac_tau,
        eps=eps,
        t_singularity=t,
    )
    jac_tau = xp.sqrt(xp.sum(dx(tau) ** 2, axis=-1))
    return (1j / 4) * h1 * jac_tau, (1j / 4) * h2 * jac_tau


def dlp_kernel_split(
    t: Array,
    tau: Array,
    k: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    ddx: Callable[[Array], Array],
    /,
    *,
    eps: float = 0.0,
) -> tuple[Array, Array]:
    r"""
    Split double-layer kernel into log-singular and analytic parts.

    Parameters
    ----------
    t : Array
        Source nodes of shape (...,).
    tau : float
        Target nodes of shape (...,).
    k : float
        Wave number of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization returning shape (..., 2).
    dx : Callable[[Array], Array]
        First derivative of the parametrization of shape (..., 2).
    ddx : Callable[[Array], Array]
        Second derivative of the parametrization of shape (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace by the diagonal limit value.

    Returns
    -------
    Array
        Log-singular coefficient of shape (N',).
    Array
        Analytic remainder of shape (N',).

    """
    xp = array_namespace(t, tau)
    x_t = x(t)

    def fval(t_in: Array) -> Array:
        diff = x_t - x(t_in)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        return k * r

    h1, h2 = hankel_h1_h2(
        tau,
        order=1,
        f=fval,
        fprime0=None,
        eps=eps,
        t_singularity=t,
    )
    d_t = D_t(
        t,
        tau,
        x,
        dx,
        ddx,
        eps=eps,
    )
    return (1j / 4) * h1 * d_t, (1j / 4) * h2 * d_t
