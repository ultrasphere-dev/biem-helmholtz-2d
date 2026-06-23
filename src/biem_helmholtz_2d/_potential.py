from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from biem_helmholtz_2d._hankel import hankel_h1_h2

from ._is_close import is_close


def A1(
    *,
    t: Array,
    tau: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    ddx: Callable[[Array], Array],
    eps: float = 0.0,
) -> Array:
    r"""
    Kernel factor $A_1(t, \tau)$ for the double-layer potential.

    $$
    A_1(t, \tau) := \frac{n^*(\tau) \cdot (x(t) - x(\tau))}{|x(t) - x(\tau)|^2}
    $$

    Parameters
    ----------
    t : Array
        Source nodes t of shape (...,).
    tau : Array
        Target nodes tau of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization x of (...,) -> (..., 2).
    dx : Callable[[Array], Array]
        First derivative dx of the parametrization of (...,) -> (..., 2).
    ddx : Callable[[Array], Array]
        Second derivative ddx of the parametrization of (...,) -> (..., 2).
    eps : float
        If ``abs(tau - t) <= eps``, replace A_1(t, tau) by the diagonal limit value.

    Returns
    -------
    Array
        Values of $A_1(t, \tau)$ of shape (...,).

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
    n_star_tau = xp.stack([dx_tau[..., 1], -dx_tau[..., 0]], axis=-1)
    result = xp.sum(n_star_tau * diff, axis=-1) / (dist**2)

    # limit part
    near0 = is_close(t, tau, eps)
    limit = (ddx_tau[..., 0] * dx_tau[..., 1] - ddx_tau[..., 1] * dx_tau[..., 0]) / (
        2 * xp.linalg.vector_norm(dx_tau, axis=-1) ** 2
    )
    return xp.where(near0, limit, result)


def slp_kernel_split(
    *,
    t: Array,
    tau: Array,
    k: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    eps: float = 0.0,
) -> tuple[Array, Array]:
    r"""
    Split single-layer kernel into log-singular and analytic parts.

    Parameters
    ----------
    t : Array
        Source nodes t of shape (...,).
    tau : float
        Target node tau of shape (...,).
    k : float
        Wave number k of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization x of (...,) -> (..., 2).
    dx : Callable[[Array], Array]
        First derivative dx of the parametrization of (...,) -> (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace the singular value of S(t, tau).

    Returns
    -------
    Array
        Log-singular coefficient of $S(t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $S(t, \tau)$ of shape (...,).

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
    *,
    t: Array,
    tau: Array,
    k: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    ddx: Callable[[Array], Array],
    eps: float = 0.0,
) -> tuple[Array, Array]:
    r"""
    Split double-layer kernel into log-singular and analytic parts.

    Parameters
    ----------
    t : Array
        Source nodes t of shape (...,).
    tau : float
        Target node tau of shape (...,).
    k : float
        Wave number k of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization x of  (...,) -> (..., 2).
    dx : Callable[[Array], Array]
        First derivative dx of the parametrization of (...,) -> (..., 2).
    ddx : Callable[[Array], Array]
        Second derivative ddx of the parametrization of (...,) -> (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace the singular value of D(t, tau).

    Returns
    -------
    Array
        Log-singular coefficient of $D(t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $D(t, \tau)$ of shape (...,).

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
    a1 = A1(
        t=t,
        tau=tau,
        x=x,
        dx=dx,
        ddx=ddx,
        eps=eps,
    )
    return (1j / 4) * h1 * a1, (1j / 4) * h2 * a1
