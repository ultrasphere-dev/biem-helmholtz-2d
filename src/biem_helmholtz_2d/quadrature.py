from __future__ import annotations

from functools import lru_cache
from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull


@lru_cache(maxsize=None)
def harmonic_number(n: int, /) -> float:
    r"""Return the harmonic number $H_n = \sum_{k=1}^n 1/k$."""
    if n < 0:
        msg = "n must be non-negative."
        raise ValueError(msg)
    if n == 0:
        return 0
    return harmonic_number(n - 1) + 1 / n


def cot_power_fourier_integral_coefficients(
    n_harmonics: int,
    power: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> Array:
    r"""
    Fourier coefficients of the finite-part integral of $\cot^{\mathrm{power}}(t/2)$.

    Returns $I_{m,\mathrm{power}}$ for $m = -(n_harmonics-1), \ldots, n_harmonics-1$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics``.
    power : int
        The exponent ``n`` in $I_{m,n}$.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Complex-valued coefficients $I_{m,\mathrm{power}}$ of shape (2*n_harmonics - 1,).

    """
    if n_harmonics <= 0:
        msg = "n_harmonics must be positive."
        raise ValueError(msg)
    if power < 0:
        msg = "power must be non-negative."
        raise ValueError(msg)

    two_pi = 2 * xp.asarray(xp.pi, dtype=dtype)

    # m = -(n_harmonics-1), ..., (n_harmonics-1)
    m = xp.arange(-(n_harmonics - 1), n_harmonics, device=device)

    # Initial values
    i0 = xp.where(m == 0, two_pi, 0)
    i1 = xp.where(m == 0, 0, two_pi * 1j * xp.sign(m))

    if power == 0:
        return i0
    if power == 1:
        return i1

    i_nm2 = i0
    i_nm1 = i1
    for k in range(2, power + 1):
        i_n = (2j * m) / (k - 1) * i_nm1 - i_nm2
        i_nm2, i_nm1 = i_nm1, i_n
    return i_nm1


def log_cot_power_fourier_integral_coefficients(
    n_harmonics: int,
    power: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> Array:
    r"""
    Fourier coefficients of the finite-part integral of
    $\log(4\sin^2(t/2))\,\cot^{\mathrm{power}}(t/2)$.

    Returns $J_{m,\mathrm{power}}$ for $m = -(n_harmonics-1), \ldots, n_harmonics-1$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics``.
    power : int
        The exponent ``n`` in $J_{m,n}$.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Complex-valued coefficients $J_{m,\mathrm{power}}$ of shape (2*n_harmonics - 1,).

    """
    if n_harmonics <= 0:
        msg = "n_harmonics must be positive."
        raise ValueError(msg)
    if power < 0:
        msg = "power must be non-negative."
        raise ValueError(msg)

    two_pi = 2 * xp.pi

    m = xp.arange(-(n_harmonics - 1), n_harmonics, device=device)
    abs_m = xp.abs(m)
    inv_abs_m = 1 / xp.astype(abs_m, dtype)

    # Initial values
    j0 = xp.where(m == 0, 0 + 0.0j, (-two_pi * inv_abs_m) + 0.0j)

    # Harmonic numbers are computed on CPU as Python scalars.
    h = xp.asarray(
        [harmonic_number(int(k)) for k in abs_m],
        device=device,
        dtype=dtype,
    )
    j1 = xp.where(
        m == 0,
        0 + 0.0j,
        two_pi * 1j * xp.sign(m) * (2 * h - inv_abs_m),
    )

    if power == 0:
        return j0
    if power == 1:
        return j1

    j_nm2 = j0
    j_nm1 = j1
    i_nm2 = cot_power_fourier_integral_coefficients(
        n_harmonics, 0, xp=xp, device=device, dtype=dtype
    )
    i_nm1 = cot_power_fourier_integral_coefficients(
        n_harmonics, 1, xp=xp, device=device, dtype=dtype
    )

    for k in range(2, power + 1):
        # Compute I_{m,k} in sync for the inhomogeneous term of J.
        i_n = (2j * m) / (k - 1) * i_nm1 - i_nm2
        j_n = (2j * m) / (k - 1) * j_nm1 - j_nm2 + (2 / (k - 1)) * i_n
        i_nm2, i_nm1 = i_nm1, i_n
        j_nm2, j_nm1 = j_nm1, j_n
    return j_nm1


def _fourier_nodes(
    n_harmonics: int,
    /,
    *,
    t_start: float = 0,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> Array:
    n_quad = 2 * n_harmonics - 1
    j = xp.astype(xp.arange(n_quad, device=device), dtype)
    return t_start + (2 * xp.pi) * j / n_quad


def cot_power_shifted_quadrature(
    n_harmonics: int,
    power: int,
    /,
    *,
    t_start: float = 0,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Shifted finite-part trapezoidal rule for $\cot^{\mathrm{power}}(t/2)$.

    Let $N' := 2 N - 1$ and $t_j := t_\mathrm{start} + 2\pi j / N'$.
    For $t_s := t_\mathrm{start}$, the rule matches the Typst statement

    $$
    \int_0^{2\pi}{}^\dash f(t)\,\cot^{\mathrm{power}}(t/2)\,dt
    = \sum_{j=0}^{N'-1} f(t_j + t_s)\,P_j^{(N',\mathrm{power})},
    $$

    with

    $$
    P_j^{(N',\mathrm{power})} := \frac{1}{N'} \sum_{|m|<N} I_{m,\mathrm{power}} e^{-i m (t_j + t_s)}.
    $$

    The returned weights correspond to $P_j^{(N',\mathrm{power})}$ evaluated at
    $t_s = t_\mathrm{start}$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics`` are integrated exactly.
    power : int
        Exponent in $\cot^{\mathrm{power}}$.
    t_start : float
        Grid shift $t_\mathrm{start}$ (sets $t_s$ in the Typst formula).
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Nodes $t_j$ of shape (2*n_harmonics - 1,).
    Array
        Weights $P_j$ of shape (2*n_harmonics - 1,).

    """
    t = _fourier_nodes(n_harmonics, t_start=t_start, xp=xp, device=device, dtype=dtype)
    n_quad = 2 * n_harmonics - 1

    coeff = cot_power_fourier_integral_coefficients(
        n_harmonics, power, xp=xp, device=device, dtype=dtype
    )
    m = xp.arange(
        -(n_harmonics - 1), n_harmonics, device=device
    )
    phase = (-1j) * m[:, None] * (t[None, :] + t_start)
    weights = xp.asarray(
        xp.real((1 / n_quad) * xp.sum(coeff[:, None] * xp.exp(phase), axis=0)),
        device=device,
        dtype=dtype,
    )
    return t, weights


def log_cot_power_shifted_quadrature(
    n_harmonics: int,
    power: int,
    /,
    *,
    t_start: float = 0,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Shifted finite-part trapezoidal rule for
    $\log(4\sin^2(t/2))\,\cot^{\mathrm{power}}(t/2)$.

    Let $N' := 2 N - 1$ and $t_j := t_\mathrm{start} + 2\pi j / N'$.
    For $t_s := t_\mathrm{start}$, the rule matches the Typst statement

    $$
    \int_0^{2\pi}{}^\dash f(t)\,\log(4\sin^2(t/2))\,\cot^{\mathrm{power}}(t/2)\,dt
    = \sum_{j=0}^{N'-1} f(t_j + t_s)\,Q_j^{(N',\mathrm{power})},
    $$

    with

    $$
    Q_j^{(N',\mathrm{power})} := \frac{1}{N'} \sum_{|m|<N} J_{m,\mathrm{power}} e^{-i m (t_j + t_s)}.
    $$

    The returned weights correspond to $Q_j^{(N',\mathrm{power})}$ evaluated at
    $t_s = t_\mathrm{start}$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics`` are integrated exactly.
    power : int
        Exponent in $\cot^{\mathrm{power}}$.
    t_start : float
        Grid shift $t_\mathrm{start}$ (sets $t_s$ in the Typst formula).
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Nodes $t_j$ of shape (2*n_harmonics - 1,).
    Array
        Weights $Q_j$ of shape (2*n_harmonics - 1,).

    """
    t = _fourier_nodes(n_harmonics, t_start=t_start, xp=xp, device=device, dtype=dtype)
    n_quad = 2 * n_harmonics - 1

    coeff = log_cot_power_fourier_integral_coefficients(
        n_harmonics, power, xp=xp, device=device, dtype=dtype
    )
    m = xp.arange(
        -(n_harmonics - 1), n_harmonics, device=device
    )
    phase = (-1j) * m[:, None] * (t[None, :] + t_start)
    weights = xp.asarray(
        xp.real((1 / n_quad) * xp.sum(coeff[:, None] * xp.exp(phase), axis=0)),
        device=device,
        dtype=dtype,
    )
    return t, weights


def trapezoidal_quadrature(
    n: int,
    /,
    *,
    t_start: float = 0,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Trapezoidal quadrature for [0, 2π].

    Returns $x_j$ and $w_j$, where

    $$
    \int_0^{2\pi} f(x) dx
    \approx \sum_{j=0}^{2n-1} w_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    t_start : float
        Grid shift $t_\mathrm{start}$, with $x_j := t_\mathrm{start} + 2\pi j / (2n-1)$.
    xp: ArrayNamespaceFull
        The array namespace.
    device: Any
        The device.
    dtype: Any
        The dtype.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $w_j$ of shape (2n - 1,).

    """
    t = _fourier_nodes(n, t_start=t_start, xp=xp, device=device, dtype=dtype)
    n_quad = t.shape[0]
    w = xp.full((1,), (2 * xp.pi) / n_quad, dtype=dtype, device=device)
    return t, w


def kussmaul_martensen_kress_quadrature(
    n: int,
    /,
    *,
    t_start: float = 0,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Kussmaul-Martensen (Kress) quadrature.

    Returns $x_j$ and $R_j$, where

    Let $n' := 2n - 1$ and $x_j := t_\mathrm{start} + 2\pi j / n'$.

    $$
    \int_0^{2\pi} \log \left(4 \sin^2 \frac{t}{2}\right) f(t) dt
    \approx \sum_{j=0}^{n'-1} R_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    t_start : float
        Grid shift $t_\mathrm{start}$.
    xp: ArrayNamespaceFull
        The array namespace.
    device: Any
        The device.
    dtype: Any
        The dtype.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $R_j$ of shape (2n - 1,).

    """
    # power == 0 corresponds to the classic Kress log quadrature.
    return log_cot_power_shifted_quadrature(
        n, 0, t_start=t_start, xp=xp, device=device, dtype=dtype
    )


def garrick_wittich_quadrature(
    n: int,
    /,
    *,
    t_start: float = 0,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Garrick-Wittich quadrature.

    Returns $x_j$ and $T_j$, where

    Let $n' := 2n - 1$ and $x_j := t_\mathrm{start} + 2\pi j / n'$.

    $$
    p.v. \int_0^{2\pi} \cot \frac{t}{2} f'(t) dt
    \approx \sum_{j=0}^{n'-1} T_j f(x_j)
    $$

    Parameters
    ----------
    n : int
        Harmonics which order is less than n are integrated exactly.
    t_start : float
        Grid shift $t_\mathrm{start}$.
    xp: ArrayNamespaceFull
        The array namespace.
    device: Any
        The device.
    dtype: Any
        The dtype.

    Returns
    -------
    Array
        The roots $x_j$ of shape (2n - 1,).
        and weights $T_j$ of shape (2n - 1,).

    """
    # power == 1 corresponds to the Cauchy principal value cot-kernel.
    return cot_power_shifted_quadrature(
        n, 1, t_start=t_start, xp=xp, device=device, dtype=dtype
    )
