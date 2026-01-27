from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from biem_helmholtz_2d.hankel import hankel_h1_h2


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
    dx_t = dx(t)
    ddx_t = ddx(t)

    diff = x_tau - x_t
    normal = xp.stack([-dx_t[..., 1], dx_t[..., 0]], axis=-1)
    numer_ = xp.sum(normal * diff, axis=-1)
    denom = xp.sum(diff**2, axis=-1)

    two_pi = 2 * xp.pi
    delta = xp.remainder(tau - t + xp.pi, two_pi) - xp.pi
    near0 = xp.abs(delta) <= eps
    denom_safe = xp.where(near0, 1, denom)
    core = numer_ / denom_safe
    limit = (dx_t[..., 1] * ddx_t[..., 0] - dx_t[..., 0] * ddx_t[..., 1]) / (
        2 * xp.sum(dx_t**2, axis=-1)
    )
    return xp.where(near0, limit, core)


def slp(
    t: Array,
    tau: float,
    k: float,
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
        Source nodes of shape (N',).
    tau : float
        Target node location.
    k : float
        Wave number.
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
    xp = array_namespace(t)
    tau_array = _asarray_like(t, tau)
    x_tau = x(tau_array)
    dx_tau = dx(tau_array)
    jac_tau = xp.sqrt(xp.sum(dx_tau**2, axis=-1))

    def fval(t_in: Array) -> Array:
        diff = x_tau - x(t_in)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        return k * r

    h1, h2 = hankel_h1_h2(
        t,
        0,
        fval,
        k * jac_tau,
        eps,
        t_singularity=tau,
    )
    jac_t = xp.sqrt(xp.sum(dx(t) ** 2, axis=-1))
    return (1j / 4) * h1 * jac_t, (1j / 4) * h2 * jac_t


def dlp(
    t: Array,
    tau: float,
    k: float,
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
        Source nodes of shape (N',).
    tau : float
        Target node location.
    k : float
        Wave number.
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
    xp = array_namespace(t)
    tau_array = _asarray_like(t, tau)
    x_tau = x(tau_array)

    def fval(t_in: Array) -> Array:
        diff = x_tau - x(t_in)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        return k * r

    h1, h2 = hankel_h1_h2(
        t,
        1,
        fval,
        None,
        eps,
        t_singularity=tau,
    )
    d_t = D_t(
        t,
        tau_array,
        x,
        dx,
        ddx,
        eps=eps,
    )
    return (1j / 4) * h1 * d_t, (1j / 4) * h2 * d_t


def _asarray_like(x: Array, value: float | Array, /) -> Array:
    xp = array_namespace(x)
    dtype = x.dtype
    device = getattr(x, "device", None)
    if device is None:
        return xp.asarray(value, dtype=dtype)
    return xp.asarray(value, device=device, dtype=dtype)
