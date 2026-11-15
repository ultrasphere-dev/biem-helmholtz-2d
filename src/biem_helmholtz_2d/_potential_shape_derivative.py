from __future__ import annotations

from array_api.latest import Array
from array_api_compat import array_namespace

from ._hankel import hankel_h1_h2
from ._is_close import is_close
from ._potential import A1


def A2(
    *,
    t: Array,
    tau: Array,
    x_t: Array,
    x_tau: Array,
    dx_tau: Array,
    h_t: Array,
    h_tau: Array,
    dh_tau: Array,
    eps: float = 0,
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
        Target nodes $t$ of shape (...,).
    tau : Array
        Source nodes $\tau$ of shape (...,).
    x_t, x_tau, dx_tau : Array
        Boundary values $x(t)$, $x(\tau)$, and $x'(\tau)$ of shape (..., 2).
    h_t, h_tau, dh_tau : Array
        Perturbation values $h(t)$, $h(\tau)$, and $h'(\tau)$ of shape (..., 2).
    eps : float
        If ``abs(tau - t) <= eps``, replace $A_2(t, \tau)$ by the diagonal limit value.

    Returns
    -------
    Array
        Values of $A_2(t, \tau)$ of shape (...,).

    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    xp = array_namespace(t, tau, x_t, x_tau, dx_tau, h_t, h_tau, dh_tau)
    diff_x = x_t - x_tau
    diff_h = h_t - h_tau
    result = xp.sum(diff_x * diff_h, axis=-1) / xp.sum(diff_x**2, axis=-1)
    limit = xp.sum(dx_tau * dh_tau, axis=-1) / xp.sum(dx_tau**2, axis=-1)
    return xp.where(is_close(t, tau, eps), limit, result)


def D4(
    *,
    t: Array,
    tau: Array,
    x_t: Array,
    x_tau: Array,
    dx_tau: Array,
    ddx_tau: Array,
    h_t: Array,
    h_tau: Array,
    dh_tau: Array,
    ddh_tau: Array,
    eps: float = 0,
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
        Target nodes $t$ of shape (...,).
    tau : Array
        Source nodes $\tau$ of shape (...,).
    x_t, x_tau, dx_tau, ddx_tau : Array
        Boundary positions and derivatives at $t$ and $\tau$ of shape (..., 2).
    h_t, h_tau, dh_tau, ddh_tau : Array
        Perturbations and derivatives at $t$ and $\tau$ of shape (..., 2).
    eps : float
        If ``abs(tau - t) <= eps``, replace $D_4(t, \tau)$ by the diagonal limit value.

    Returns
    -------
    Array
        Values of $D_4(t, \tau)$ of shape (...,).

    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    xp = array_namespace(t, tau, x_t, x_tau, dx_tau, ddx_tau, h_t, h_tau, dh_tau, ddh_tau)
    diff_x = x_t - x_tau
    diff_h = h_t - h_tau
    normal = xp.stack([dx_tau[..., 1], -dx_tau[..., 0]], axis=-1)
    normal_h = xp.stack([dh_tau[..., 1], -dh_tau[..., 0]], axis=-1)
    result = (xp.sum(normal * diff_h, axis=-1) + xp.sum(normal_h * diff_x, axis=-1)) / xp.sum(
        diff_x**2, axis=-1
    )
    numerator = (
        dh_tau[..., 1] * ddx_tau[..., 0]
        - dh_tau[..., 0] * ddx_tau[..., 1]
        + dx_tau[..., 1] * ddh_tau[..., 0]
        - dx_tau[..., 0] * ddh_tau[..., 1]
    )
    limit = numerator / (2 * xp.sum(dx_tau**2, axis=-1))
    return xp.where(is_close(t, tau, eps), limit, result)


def slp_shape_derivative_split(
    *,
    t: Array,
    tau: Array,
    k: Array,
    x_t: Array,
    x_tau: Array,
    dx_tau: Array,
    h_t: Array,
    h_tau: Array,
    dh_tau: Array,
    eps: float = 0,
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
        Target nodes $t$ of shape (...,).
    tau : Array
        Source nodes $\tau$ of shape (...,).
    k : Array
        Wave number $k$ of shape (...,).
    x_t, x_tau, dx_tau : Array
        Boundary values $x(t)$, $x(\tau)$, and $x'(\tau)$ of shape (..., 2).
    h_t, h_tau, dh_tau : Array
        Perturbation values $h(t)$, $h(\tau)$, and $h'(\tau)$ of shape (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace singular values by their diagonal limits.

    Returns
    -------
    Array
        Log-singular coefficient of $S'[h](t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $S'[h](t, \tau)$ of shape (...,).

    """
    xp = array_namespace(t, tau, k, x_t, x_tau, dx_tau, h_t, h_tau, dh_tau)
    jac = xp.linalg.vector_norm(dx_tau, axis=-1)
    argument = k * xp.linalg.vector_norm(x_t - x_tau, axis=-1)
    h1_1, h2_1 = hankel_h1_h2(tau, order=1, f=argument, fprime0=None, eps=eps, t_singularity=t)
    h1_0, h2_0 = hankel_h1_h2(tau, order=0, f=argument, fprime0=k * jac, eps=eps, t_singularity=t)
    a2 = A2(
        t=t,
        tau=tau,
        x_t=x_t,
        x_tau=x_tau,
        dx_tau=dx_tau,
        h_t=h_t,
        h_tau=h_tau,
        dh_tau=dh_tau,
        eps=eps,
    )
    x_dot_h = xp.sum(dx_tau * dh_tau, axis=-1)
    return (
        (1j / 4) * (-h1_1 * a2 * jac + h1_0 * x_dot_h / jac),
        (1j / 4) * (-h2_1 * a2 * jac + h2_0 * x_dot_h / jac),
    )


def dlp_shape_derivative_split(
    *,
    t: Array,
    tau: Array,
    k: Array,
    x_t: Array,
    x_tau: Array,
    dx_tau: Array,
    ddx_tau: Array,
    h_t: Array,
    h_tau: Array,
    dh_tau: Array,
    ddh_tau: Array,
    eps: float = 0,
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
        Target nodes $t$ of shape (...,).
    tau : Array
        Source nodes $\tau$ of shape (...,).
    k : Array
        Wave number $k$ of shape (...,).
    x_t, x_tau, dx_tau, ddx_tau : Array
        Boundary positions and derivatives at $t$ and $\tau$ of shape (..., 2).
    h_t, h_tau, dh_tau, ddh_tau : Array
        Perturbations and derivatives at $t$ and $\tau$ of shape (..., 2).
    eps : float
        If ``abs(t - tau) <= eps``, replace singular values by their diagonal limits.

    Returns
    -------
    Array
        Log-singular coefficient of $D'[h](t, \tau)$ of shape (...,).
    Array
        Analytic remainder of $D'[h](t, \tau)$ of shape (...,).

    """
    xp = array_namespace(t, tau, k, x_t, x_tau, dx_tau, ddx_tau, h_t, h_tau, dh_tau, ddh_tau)
    argument = k * xp.linalg.vector_norm(x_t - x_tau, axis=-1)
    h1_2, h2_2 = hankel_h1_h2(tau, order=2, f=argument, fprime0=None, eps=eps, t_singularity=t)
    h1_1, h2_1 = hankel_h1_h2(tau, order=1, f=argument, fprime0=None, eps=eps, t_singularity=t)
    a1 = A1(t=t, tau=tau, x_t=x_t, x_tau=x_tau, dx_tau=dx_tau, ddx_tau=ddx_tau, eps=eps)
    a2 = A2(
        t=t,
        tau=tau,
        x_t=x_t,
        x_tau=x_tau,
        dx_tau=dx_tau,
        h_t=h_t,
        h_tau=h_tau,
        dh_tau=dh_tau,
        eps=eps,
    )
    d4 = D4(
        t=t,
        tau=tau,
        x_t=x_t,
        x_tau=x_tau,
        dx_tau=dx_tau,
        ddx_tau=ddx_tau,
        h_t=h_t,
        h_tau=h_tau,
        dh_tau=dh_tau,
        ddh_tau=ddh_tau,
        eps=eps,
    )
    return (1j / 4) * (-h1_2 * a2 * a1 + h1_1 * d4), (1j / 4) * (-h2_2 * a2 * a1 + h2_1 * d4)
