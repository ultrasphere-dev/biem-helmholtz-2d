from __future__ import annotations

import math
from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from ._is_close import periodic_difference
from ._scipy_wrapper import scipy_jv, scipy_yv

_EULER_MASCHERONI: float = 0.57721566490153286060651209008240243104215933593992


def _scipy_jv_yv(
    order: int,
    x: Array,
    /,
) -> tuple[Array, Array]:
    xp = array_namespace(x)
    j = scipy_jv(order, x)
    y = scipy_yv(order, x)
    return j, y


def _neumann_y1_y2(
    x: Array,
    /,
    *,
    order: int,
    f: Callable[[Array], Array],
    fprime0: Array | None = None,
    eps: float = 0.0,
) -> tuple[Array, Array]:
    if order == 0 and fprime0 is None:
        msg = "fprime0 (shape (...,)) is required when order == 0."
        raise ValueError(msg)

    fx = f(x)
    print(x, fx)
    xp = array_namespace(x, fx, fprime0)
    jv, yv = _scipy_jv_yv(order, fx)

    if order == 0:
        fx_pow = 1
    else:
        fx_pow = fx**order

    if eps < 0:
        msg = "eps must be non-negative."
        raise ValueError(msg)

    y1 = fx_pow * jv / xp.pi
    zero = xp.asarray(0, device=x.device, dtype=x.dtype)
    delta = periodic_difference(zero, x)
    log_kernel = xp.log(4 * xp.sin(delta / 2) ** 2)
    y2 = fx_pow * yv - y1 * log_kernel

    near0 = xp.abs(delta) <= eps
    if order == 0:
        assert fprime0 is not None
        y2_lim = (2 / xp.pi) * (
            xp.log(xp.abs(fprime0[(None,) * x.ndim + (...,)]) / 2) + _EULER_MASCHERONI
        )
    else:
        y2_lim = -((2**order) * math.factorial(order - 1)) / xp.pi

    y2 = xp.where(near0, y2_lim, y2)
    return y1, y2


def neumann_y1_y2(
    x: Array,
    /,
    *,
    order: int,
    f: Callable[[Array], Array],
    fprime0: Array | None = None,
    eps: float = 0.0,
    t_singularity: Array,
) -> tuple[Array, Array]:
    r"""
    Split Neumann functions into log-singular and analytic parts on nodes ``x``.

    The split is

    $$
    f(x)^{\mathrm{order}} Y_{\mathrm{order}}(f(x))
    = Y_{\mathrm{order}}^{(1)}(x)
    \,\log\left(4\sin^2\frac{x - t_s}{2}\right) + Y_{\mathrm{order}}^{(2)}(x).
    $$

    Parameters
    ----------
    x : Array
        Points to evaluate of shape (...x).
    order : int
        Order of the Neumann function.
    f : Callable[[Array], Array]
        Function evaluated at nodes. It must accept input of shape (...x)
        and return an array of shape (...x, ...f). It is assumed to be smooth
        everywhere with $f(t_s) = 0$ and $f'(t_s) \ne 0$.
    fprime0 : Array | None
        Value $f'(t_s)$ of shape (...f) required when ``order == 0``.
    eps : float
        If ``abs(x - t_s) <= eps``, replace $Y^{(2)}$ by its limit value.
    t_singularity : Array
        Singularity locations $t_s$ of shape (...f).

    Returns
    -------
    Array
        $Y^{(1)}$ of shape (...x, ...f).
    Array
        $Y^{(2)}$ of shape (...x, ...f).

    """
    xp = array_namespace(x, t_singularity)
    t_s_arr = xp.asarray(t_singularity, device=x.device, dtype=x.dtype)
    t_s_view = t_s_arr[(None,) * x.ndim + (...,)]
    x_view = x[(...,) + (None,) * t_s_arr.ndim]
    x_shifted = x_view - t_s_view

    def f_shifted(x_in: Array) -> Array:
        xp_local = array_namespace(x_in, t_s_arr)
        return f(x_in + t_s_view)

    return _neumann_y1_y2(
        x_shifted,
        order=order,
        f=f_shifted,
        fprime0=fprime0,
        eps=eps,
    )


def hankel_h1_h2(
    x: Array,
    /,
    *,
    order: int,
    f: Callable[[Array], Array],
    fprime0: Array | None = None,
    eps: float = 0.0,
    t_singularity: Array,
) -> tuple[Array, Array]:
    r"""
    Split Hankel functions of the first kind into log-singular and analytic parts.

    The split is

    $$
    f(x)^{\mathrm{order}} H_{\mathrm{order}}^{(1)}(f(x))
    = H_{\mathrm{order}}^{(1,1)}(x)\,\log\left(4\sin^2\frac{x - t_s}{2}\right)
    + H_{\mathrm{order}}^{(1,2)}(x).
    $$

    Parameters
    ----------
    x : Array
        Points to evaluate of shape (...x).
    order : int
        Order of the Hankel function.
    f : Callable[[Array], Array]
        Function evaluated at nodes. It must accept input of shape (...x)
        and return an array of shape (...x, ...f). It is assumed to be smooth
        everywhere with $f(t_s) = 0$ and $f'(t_s) \ne 0$.
    fprime0 : Array | None
        Value $f'(t_s)$ of shape (...f) required when ``order == 0``.
    eps : float
        If ``abs(x - t_s) <= eps``, replace $H^{(1,2)}$ by its limit value.
    t_singularity : Array
        Singularity locations $t_s$ of shape (...f).

    Returns
    -------
    Array
        $H^{(1,1)}$ of shape (...x, ...f).
    Array
        $H^{(1,2)}$ of shape (...x, ...f).

    """
    y1, y2 = neumann_y1_y2(
        x,
        order=order,
        f=f,
        fprime0=fprime0,
        eps=eps,
        t_singularity=t_singularity,
    )
    xp = array_namespace(x, y1, y2)
    h1 = 1j * y1
    h2 = (xp.pi * y1) + (1j * y2)
    return h1, h2
