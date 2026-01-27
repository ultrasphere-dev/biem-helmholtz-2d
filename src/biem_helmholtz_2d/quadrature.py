from __future__ import annotations

import math
from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull

from ._integral import (
    cot_power_fourier_integral_coefficients,
    log_cot_power_fourier_integral_coefficients,
)


def _resolve_t_start(
    n_harmonics: int,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> float:
    if t_start_factor is not None and t_start is not None:
        msg = "Specify only one of t_start or t_start_factor."
        raise ValueError(msg)
    if t_start_factor is None:
        if t_start is None:
            return 0
        return t_start
    h = (2 * math.pi) / (2 * n_harmonics - 1)
    return t_start_factor * h


def trapezoidal_quadrature(
    n: int,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
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
    t_start : float | None
        Grid shift $t_\mathrm{start}$, with $x_j := t_\mathrm{start} + 2\pi j / (2n-1)$.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$. Mutually exclusive with
        ``t_start``.
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
    t_start = _resolve_t_start(n, t_start=t_start, t_start_factor=t_start_factor)
    n_quad = 2 * n - 1
    j = xp.astype(xp.arange(n_quad, device=device), dtype)
    t = t_start + (2 * xp.pi) * j / n_quad
    w = xp.full((1,), (2 * xp.pi) / n_quad, dtype=dtype, device=device)
    return t, w


def fourier_coeff_to_quadrature(
    coeff: Array,
    n_harmonics: int,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Build quadrature nodes and weights from Fourier integral coefficients.

    Parameters
    ----------
    coeff : Array
        Fourier coefficients $I_m$ of shape (2*n_harmonics - 1,).
    n_harmonics : int
        Harmonics with order less than ``n_harmonics`` are integrated exactly.
    t_start : float | None
        Grid shift $t_\mathrm{start}$ (sets $t_s$ in the Fourier sum).
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n_harmonics-1)$. Mutually exclusive with
        ``t_start``.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Nodes $t_j + t_\mathrm{start}$ of shape (2*n_harmonics - 1,).
    Array
        Weights derived from ``coeff`` of shape (2*n_harmonics - 1,).
    """
    t, _ = trapezoidal_quadrature(
        n_harmonics,
        t_start=t_start,
        t_start_factor=t_start_factor,
        xp=xp,
        device=device,
        dtype=dtype,
    )
    n_quad = t.shape[0]
    m = xp.arange(-(n_harmonics - 1), n_harmonics, device=device)
    phase = (-1j) * m[:, None] * t[None, :]
    weights = xp.asarray(
        xp.real((1 / n_quad) * xp.sum(coeff[:, None] * xp.exp(phase), axis=0)),
        device=device,
        dtype=dtype,
    )
    return t, weights


def cot_power_quadrature(
    n_harmonics: int,
    power: int,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Shifted finite-part trapezoidal rule for $\cot^{\mathrm{power}}(t/2)$.

    Let $N' := 2 N - 1$ and $t_j := 2\pi j / N'$.
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
    $t_s = t_\mathrm{start}$, and the returned nodes are $t_j + t_\mathrm{start}$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics`` are integrated exactly.
    power : int
        Exponent in $\cot^{\mathrm{power}}$.
    t_start : float | None
        Grid shift $t_\mathrm{start}$ (sets $t_s$ in the Typst formula).
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$. Mutually exclusive with
        ``t_start``.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Nodes $t_j + t_\mathrm{start}$ of shape (2*n_harmonics - 1,).
    Array
        Weights $P_j$ of shape (2*n_harmonics - 1,).

    """
    coeff = cot_power_fourier_integral_coefficients(
        n_harmonics, power, xp=xp, device=device, dtype=dtype
    )
    return fourier_coeff_to_quadrature(
        coeff,
        n_harmonics,
        t_start=t_start,
        t_start_factor=t_start_factor,
        xp=xp,
        device=device,
        dtype=dtype,
    )


def log_cot_power_quadrature(
    n_harmonics: int,
    power: int,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> tuple[Array, Array]:
    r"""
    Shifted finite-part trapezoidal rule for
    $\log(4\sin^2(t/2))\,\cot^{\mathrm{power}}(t/2)$.

    Let $N' := 2 N - 1$ and $t_j := 2\pi j / N'$.
    For $t_s := t_\mathrm{start}$, the rule matches the Typst statement

    $$
    \int_0^{2\pi}{}^\dash f(t)\,\log(4\sin^2(t/2))\,\cot^{\mathrm{power}}(t/2)\,dt
    = \sum_{j=0}^{N'-1} f(t_j + t_s)\,Q_j^{(N',\mathrm{power})},
    $$

    with

    $$
    Q_j^{(N',\mathrm{power})}
    := \frac{1}{N'} \sum_{|m|<N} J_{m,\mathrm{power}} e^{-i m (t_j + t_s)}.
    $$

    The returned weights correspond to $Q_j^{(N',\mathrm{power})}$ evaluated at
    $t_s = t_\mathrm{start}$, and the returned nodes are $t_j + t_\mathrm{start}$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics`` are integrated exactly.
    power : int
        Exponent in $\cot^{\mathrm{power}}$.
    t_start : float | None
        Grid shift $t_\mathrm{start}$ (sets $t_s$ in the Typst formula).
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$. Mutually exclusive with
        ``t_start``.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Nodes $t_j + t_\mathrm{start}$ of shape (2*n_harmonics - 1,).
    Array
        Weights $Q_j$ of shape (2*n_harmonics - 1,).

    """
    coeff = log_cot_power_fourier_integral_coefficients(
        n_harmonics, power, xp=xp, device=device, dtype=dtype
    )
    return fourier_coeff_to_quadrature(
        coeff,
        n_harmonics,
        t_start=t_start,
        t_start_factor=t_start_factor,
        xp=xp,
        device=device,
        dtype=dtype,
    )


def kussmaul_martensen_kress_quadrature(
    n: int,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
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
    t_start : float | None
        Grid shift $t_\mathrm{start}$.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$. Mutually exclusive with
        ``t_start``.
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
    return log_cot_power_quadrature(
        n,
        0,
        t_start=t_start,
        t_start_factor=t_start_factor,
        xp=xp,
        device=device,
        dtype=dtype,
    )


def garrick_wittich_quadrature(
    n: int,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
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
    t_start : float | None
        Grid shift $t_\mathrm{start}$.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$. Mutually exclusive with
        ``t_start``.
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
    return cot_power_quadrature(
        n,
        1,
        t_start=t_start,
        t_start_factor=t_start_factor,
        xp=xp,
        device=device,
        dtype=dtype,
    )
