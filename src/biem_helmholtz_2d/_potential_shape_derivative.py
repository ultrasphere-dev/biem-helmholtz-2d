from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from biem_helmholtz_2d._hankel import hankel_h1_h2
from biem_helmholtz_2d._is_close import is_close
from biem_helmholtz_2d._potential import A1 as _A1


def A2(
    *,
    t: Array,
    tau: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    h: Callable[[Array], Array],
    dh: Callable[[Array], Array],
    eps: float = 0.0,
) -> Array:
    r"""
    Kernel factor $A_2(t, \tau)$ for the shape derivative.

    $$
    A_2(t, \tau) := \frac{(x(t) - x(\tau)) \cdot (h(t) - h(\tau))}{|x(t) - x(\tau)|^2}
    \xrightarrow{\tau \to t}
    \frac{x'(t) \cdot h'(t)}{|x'(t)|^2}
    $$

    Parameters
    ----------
    t : Array
        Source nodes $t$ of shape (...,).
    tau : Array
        Target nodes $\tau$ of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization $x$ of (...,) -> (..., 2).
    dx : Callable[[Array], Array]
        First derivative $x'$ of the parametrization of (...,) -> (..., 2).
    h : Callable[[Array], Array]
        Perturbation $h$ of (...,) -> (..., 2).
    dh : Callable[[Array], Array]
        First derivative $h'$ of the perturbation of (...,) -> (..., 2).
    eps : float
        If ``abs(tau - t) <= eps``, replace $A_2(t, \tau)$ by the diagonal limit value.

    Returns
    -------
    Array
        Values of $A_2(t, \tau)$ of shape (...,).

    """
    if eps < 0:
        msg = "eps must be non-negative."
        raise ValueError(msg)

    xp = array_namespace(t, tau)
    x_t = x(t)
    x_tau = x(tau)
    h_t = h(t)
    h_tau = h(tau)
    dx_tau = dx(tau)
    dh_tau = dh(tau)

    # non-limit part
    diff_x = x_t - x_tau
    diff_h = h_t - h_tau
    dist_sq = xp.sum(diff_x**2, axis=-1)
    result = xp.sum(diff_x * diff_h, axis=-1) / dist_sq

    # limit part
    near0 = is_close(t, tau, eps)
    limit = xp.sum(dx_tau * dh_tau, axis=-1) / xp.sum(dx_tau**2, axis=-1)
    return xp.where(near0, limit, result)


def D4(
    *,
    t: Array,
    tau: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    ddx: Callable[[Array], Array],
    h: Callable[[Array], Array],
    dh: Callable[[Array], Array],
    ddh: Callable[[Array], Array],
    eps: float = 0.0,
) -> Array:
    r"""
    Kernel factor $D_4(t, \tau)$ for the shape derivative of the double-layer potential.

    $$
    D_4(t, \tau) := \frac{
        n^*(\tau) \cdot (h(t) - h(\tau)) + (n^*)'[h](\tau) \cdot (x(t) - x(\tau))
    }{
        |x(t) - x(\tau)|^2
    }
    $$

    where
    $$
    n^*(\tau) = (x'_2(\tau), -x'_1(\tau)), \quad
    (n^*)'[h](\tau) = (h'_2(\tau), -h'_1(\tau)).
    $$

    The diagonal limit is

    $$
    \xrightarrow{\tau \to t}
    \frac{
        (h'_2(t) x''_1(t) - h'_1(t) x''_2(t))
        + (x'_2(t) h''_1(t) - x'_1(t) h''_2(t))
    }{
        2 |x'(t)|^2
    }.
    $$

    Parameters
    ----------
    t : Array
        Source nodes $t$ of shape (...,).
    tau : Array
        Target nodes $\tau$ of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization $x$ of (...,) -> (..., 2).
    dx : Callable[[Array], Array]
        First derivative $x'$ of the parametrization of (...,) -> (..., 2).
    ddx : Callable[[Array], Array]
        Second derivative $x''$ of the parametrization of (...,) -> (..., 2).
    h : Callable[[Array], Array]
        Perturbation $h$ of (...,) -> (..., 2).
    dh : Callable[[Array], Array]
        First derivative $h'$ of the perturbation of (...,) -> (..., 2).
    ddh : Callable[[Array], Array]
        Second derivative $h''$ of the perturbation of (...,) -> (..., 2).
    eps : float
        If ``abs(tau - t) <= eps``, replace $D_4(t, \tau)$ by the diagonal limit value.

    Returns
    -------
    Array
        Values of $D_4(t, \tau)$ of shape (...,).

    """
    if eps < 0:
        msg = "eps must be non-negative."
        raise ValueError(msg)

    xp = array_namespace(t, tau)
    x_t = x(t)
    x_tau = x(tau)
    h_t = h(t)
    h_tau = h(tau)
    dx_tau = dx(tau)
    dh_tau = dh(tau)

    diff_x = x_t - x_tau
    diff_h = h_t - h_tau
    dist_sq = xp.sum(diff_x**2, axis=-1)

    n_star_tau = xp.stack([dx_tau[..., 1], -dx_tau[..., 0]], axis=-1)
    n_star_prime_tau = xp.stack([dh_tau[..., 1], -dh_tau[..., 0]], axis=-1)

    result = (
        xp.sum(n_star_tau * diff_h, axis=-1) + xp.sum(n_star_prime_tau * diff_x, axis=-1)
    ) / dist_sq

    # limit part
    near0 = is_close(t, tau, eps)
    ddx_tau = ddx(tau)
    ddh_tau = ddh(tau)
    numerator = (
        dh_tau[..., 1] * ddx_tau[..., 0]
        - dh_tau[..., 0] * ddx_tau[..., 1]
        + dx_tau[..., 1] * ddh_tau[..., 0]
        - dx_tau[..., 0] * ddh_tau[..., 1]
    )
    limit = numerator / (2 * xp.sum(dx_tau**2, axis=-1))
    return xp.where(near0, limit, result)


def slp_shape_derivative_split(
    *,
    t: Array,
    tau: Array,
    k: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    h: Callable[[Array], Array],
    dh: Callable[[Array], Array],
    eps: float = 0.0,
) -> tuple[Array, Array]:
    r"""
    Split the shape derivative of the single-layer kernel into log-singular and analytic parts.

    $$
    S'[h](t, \tau) = \frac{i}{4} \Big[
        -\mathcal{H}_1^{(1)}(k |x(t) - x(\tau)|)
        \,A_2(t, \tau)\,
        |x'(\tau)|
        + H_0^{(1)}(k |x(t) - x(\tau)|)
        \,\frac{x'(\tau) \cdot h'(\tau)}{|x'(\tau)|}
    \Big]
    $$

    where $\mathcal{H}_1^{(1)}(z) := z H_1^{(1)}(z)$.

    Parameters
    ----------
    t : Array
        Source nodes $t$ of shape (...,).
    tau : Array
        Target nodes $\tau$ of shape (...,).
    k : Array
        Wave number $k$ of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization $x$ of (...,) -> (..., 2).
    dx : Callable[[Array], Array]
        First derivative $x'$ of the parametrization of (...,) -> (..., 2).
    h : Callable[[Array], Array]
        Perturbation $h$ of (...,) -> (..., 2).
    dh : Callable[[Array], Array]
        First derivative $h'$ of the perturbation of (...,) -> (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace singular values by their diagonal limits.

    Returns
    -------
    Array
        Log-singular coefficient of $S'[h](t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $S'[h](t, \tau)$ of shape (...,).

    """
    xp = array_namespace(t, tau)
    x_t = x(t)
    dx_tau = dx(tau)
    dh_tau = dh(tau)
    jac_tau = xp.sqrt(xp.sum(dx_tau**2, axis=-1))

    def fval(t_in: Array) -> Array:
        diff = x_t - x(t_in)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        return k * r

    # order-1 Hankel: gives H_1^{(1,1)} * log + H_1^{(1,2)} = (k*r) * H_1^{(1)}(k*r)
    h1_1, h2_1 = hankel_h1_h2(
        tau,
        order=1,
        f=fval,
        fprime0=None,
        eps=eps,
        t_singularity=t,
    )

    # order-0 Hankel: gives H_0^{(1,1)} * log + H_0^{(1,2)} = H_0^{(1)}(k*r)
    h1_0, h2_0 = hankel_h1_h2(
        tau,
        order=0,
        f=fval,
        fprime0=k * jac_tau,
        eps=eps,
        t_singularity=t,
    )

    a2 = A2(t=t, tau=tau, x=x, dx=dx, h=h, dh=dh, eps=eps)
    x_dot_h = xp.sum(dx_tau * dh_tau, axis=-1)

    log_sing = (1j / 4) * (-h1_1 * a2 * jac_tau + h1_0 * x_dot_h / jac_tau)
    analytic = (1j / 4) * (-h2_1 * a2 * jac_tau + h2_0 * x_dot_h / jac_tau)
    return log_sing, analytic


def dlp_shape_derivative_split(
    *,
    t: Array,
    tau: Array,
    k: Array,
    x: Callable[[Array], Array],
    dx: Callable[[Array], Array],
    ddx: Callable[[Array], Array],
    h: Callable[[Array], Array],
    dh: Callable[[Array], Array],
    ddh: Callable[[Array], Array],
    eps: float = 0.0,
) -> tuple[Array, Array]:
    r"""
    Split the shape derivative of the double-layer kernel into log-singular and analytic parts.

    $$
    D'[h](t, \tau) = \frac{i}{4} \Big[
        -\mathcal{H}_2^{(1)}(k |x(t) - x(\tau)|)
        \,A_2(t, \tau)\,
        A_1(t, \tau)
        + \mathcal{H}_1^{(1)}(k |x(t) - x(\tau)|)
        \,D_4(t, \tau)
    \Big]
    $$

    where $\mathcal{H}_n^{(1)}(z) := z^n H_n^{(1)}(z)$.

    Parameters
    ----------
    t : Array
        Source nodes $t$ of shape (...,).
    tau : Array
        Target nodes $\tau$ of shape (...,).
    k : Array
        Wave number $k$ of shape (...,).
    x : Callable[[Array], Array]
        Boundary parametrization $x$ of (...,) -> (..., 2).
    dx : Callable[[Array], Array]
        First derivative $x'$ of the parametrization of (...,) -> (..., 2).
    ddx : Callable[[Array], Array]
        Second derivative $x''$ of the parametrization of (...,) -> (..., 2).
    h : Callable[[Array], Array]
        Perturbation $h$ of (...,) -> (..., 2).
    dh : Callable[[Array], Array]
        First derivative $h'$ of the perturbation of (...,) -> (..., 2).
    ddh : Callable[[Array], Array]
        Second derivative $h''$ of the perturbation of (...,) -> (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace singular values by their diagonal limits.

    Returns
    -------
    Array
        Log-singular coefficient of $D'[h](t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $D'[h](t, \tau)$ of shape (...,).

    """
    xp = array_namespace(t, tau)
    x_t = x(t)

    def fval(t_in: Array) -> Array:
        diff = x_t - x(t_in)
        r = xp.sqrt(xp.sum(diff**2, axis=-1))
        return k * r

    # order-2 Hankel: gives H_2^{(1,1)} * log + H_2^{(1,2)} = (k*r)^2 * H_2^{(1)}(k*r)
    h1_2, h2_2 = hankel_h1_h2(
        tau,
        order=2,
        f=fval,
        fprime0=None,
        eps=eps,
        t_singularity=t,
    )

    # order-1 Hankel: gives H_1^{(1,1)} * log + H_1^{(1,2)} = (k*r) * H_1^{(1)}(k*r)
    h1_1, h2_1 = hankel_h1_h2(
        tau,
        order=1,
        f=fval,
        fprime0=None,
        eps=eps,
        t_singularity=t,
    )

    a1 = _A1(t=t, tau=tau, x=x, dx=dx, ddx=ddx, eps=eps)
    a2 = A2(t=t, tau=tau, x=x, dx=dx, h=h, dh=dh, eps=eps)
    d4 = D4(t=t, tau=tau, x=x, dx=dx, ddx=ddx, h=h, dh=dh, ddh=ddh, eps=eps)

    log_sing = (1j / 4) * (-h1_2 * a2 * a1 + h1_1 * d4)
    analytic = (1j / 4) * (-h2_2 * a2 * a1 + h2_1 * d4)
    return log_sing, analytic
