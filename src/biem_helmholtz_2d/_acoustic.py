from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
from array_api.latest import Array
from array_api_compat import array_namespace
from ie_circle import Shapes, nystrom, trapezoidal_quadrature
from ie_circle._bie import NystromInterpolant, QuadratureType
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import quantile

from ._potential import dlp_kernel_split, slp_kernel_split
from ._potential_inner import dlp, dlp_kernel, slp, slp_kernel


def scattering_dirichlet(
    *,
    k: Array,
    shapes: Shapes,
    incident_field: Callable[[Array], Array],
    alpha: Array,
    eta: Array,
    n: int,
) -> NystromInterpolant:
    r"""
    Solve the combined field BIE for multiple scatterers.

    For $M$ scatterers with boundaries $\Gamma_1, \ldots, \Gamma_M$,
    solves the block system

    $$
    \left(\frac{\alpha_j}{2} + \alpha_j \mathcal{D}_{\Gamma_j}
    - i\eta_j \mathcal{S}_{\Gamma_j}\right) \phi_j
    + \sum_{l \neq j}
    (\alpha_l \mathcal{D}_{\Gamma_l} - i\eta_l \mathcal{S}_{\Gamma_l})
    \phi_l\big|_{\Gamma_j}
    = -u^{\text{inc}}\big|_{\Gamma_j}
    $$

    Parameters
    ----------
    k : Array
        The wave number of shape (...,).
    shapes : Shapes
        The shapes of the $M$ scatterers.
    incident_field : Callable[[Array], Array]
        The incident field of (..., 2) -> (...,).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (M,).
    eta : Array
        The coupling parameter for the single-layer potential of shape (M,).
    n : int
        The maximum order - 1.

    Returns
    -------
    NystromInterpolant
        The density with $M$ components. ``density(t)`` returns shape ``(..., M)``.

    """
    M = shapes.n_shapes
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device

    def k_log(t: Array, tau: Array) -> Array:
        # Diagonal blocks only (log-singular part)
        # t: (Q, 1), tau: (1, Q)
        diag = []
        for j in range(M):
            shape_j = shapes[j]
            slp_log, _ = slp_kernel_split(
                t=t, tau=tau, k=k[..., None, None], x=shape_j.x, dx=shape_j.dx
            )
            dlp_log, _ = dlp_kernel_split(
                t=t,
                tau=tau,
                k=k[..., None, None],
                x=shape_j.x,
                dx=shape_j.dx,
                ddx=shape_j.ddx,
            )
            diag.append(alpha[j] * dlp_log - 1j * eta[j] * slp_log)
        # stack: (..., Q, Q, M) -> (..., Q, Q, M, M) via eye
        stacked = xp.stack(diag, axis=-1)
        eye = xp.eye(M, dtype=dtype, device=device)
        return stacked[..., :, None] * eye

    def k_cont(t: Array, tau: Array) -> Array:
        # All blocks (smooth/analytic part)
        # t: (Q, 1), tau: (1, Q)
        blocks = []
        for j in range(M):
            row = []
            shape_j = shapes[j]
            xj_t = shape_j.x(t)  # (Q, 1, 2)
            for m in range(M):
                shape_m = shapes[m]
                if j == m:
                    _, slp_cont = slp_kernel_split(
                        t=t, tau=tau, k=k[..., None, None], x=shape_j.x, dx=shape_j.dx
                    )
                    _, dlp_cont = dlp_kernel_split(
                        t=t,
                        tau=tau,
                        k=k[..., None, None],
                        x=shape_j.x,
                        dx=shape_j.dx,
                        ddx=shape_j.ddx,
                    )
                    row.append(alpha[j] * dlp_cont - 1j * eta[j] * slp_cont)
                else:
                    dlp_val = dlp_kernel(xj_t, shape_x=shape_m.x, shape_dx=shape_m.dx, k=k, tau=tau)
                    slp_val = slp_kernel(xj_t, shape_x=shape_m.x, shape_dx=shape_m.dx, k=k, tau=tau)
                    row.append(alpha[m] * dlp_val - 1j * eta[m] * slp_val)
            blocks.append(xp.stack(row, axis=-1))
        return xp.stack(blocks, axis=-2)

    def a(t: Array) -> Array:
        xp = array_namespace(t)
        return xp.ones_like(t)[..., None] * (alpha / 2)

    def rhs(t: Array) -> Array:
        # shapes.x(t) returns (..., M, 2)
        x_t = shapes.x(t)  # (Q, M, 2)
        u_inc = incident_field(x_t)  # (Q, M)
        return -u_inc

    kernels = {
        (QuadratureType.NO_SINGULARITY, 0): k_cont,
        (QuadratureType.LOG_COT_POWER, 0): k_log,
    }

    return nystrom(a, kernels, rhs, n=n, xp=xp, device=device, dtype=dtype)


def _isin_shape(x: Array, shapes: Shapes, /, n_quad: int, tol: float = 1e-5) -> Array:
    """Return True for each (point, shape) pair if point is inside shape (winding number test)."""
    xp = array_namespace(x)
    t, w = trapezoidal_quadrature(n_quad, xp=xp, device=x.device, dtype=x.dtype)
    x_t = shapes.x(t)  # (Q, M, 2)
    dx_t = shapes.dx(t)  # (Q, M, 2)
    nx_t_unnormalized = xp.stack([dx_t[..., 1], -dx_t[..., 0]], axis=-1)  # (Q, M, 2)
    # x has shape (..., 2), we want output shape (..., M)
    M = shapes.n_shapes
    results = []
    for j in range(M):
        x_t_j = x_t[:, j, :]  # (Q, 2)
        nx_t_j = nx_t_unnormalized[:, j, :]  # (Q, 2)
        # x_t_j: (Q, 2), x: (..., 2) -> x[..., None, :]: (..., 1, 2)
        # diff: (..., Q, 2) after broadcasting
        diff = x_t_j - x[..., None, :]
        lower = xp.sum(diff**2, axis=-1)  # (..., Q)
        upper = xp.sum(nx_t_j * diff, axis=-1)  # (..., Q)
        integrand = upper / lower  # (..., Q)
        integral = xp.sum(integrand * w, axis=-1)  # (...)
        winding_number = integral / (2 * math.pi)  # (...)
        results.append(xp.abs(winding_number) > tol)  # (...)
    return xp.stack(results, axis=-1)  # (..., M)


def isin_shapes(x: Array, shapes: Shapes, /, n_quad: int, tol: float = 1e-5) -> Array:
    """Return True if x is inside any of the shapes."""
    xp = array_namespace(x)
    inside_per_shape = _isin_shape(x, shapes, n_quad=n_quad, tol=tol)  # (..., M)
    return xp.any(inside_per_shape, axis=-1)  # (...,)


FieldKind = Literal["uin", "uscat", "utot"]
FieldComponent = Literal["re", "im", "abs"]
FieldEntry = Mapping[str, Any]
FieldData = Mapping[FieldKind, Mapping[FieldComponent, FieldEntry]]

_FIELD_NAMES: dict[FieldKind, str] = {
    "uin": "Incident",
    "uscat": "Scattered",
    "utot": "Total",
}
_COMPONENT_NAMES: dict[FieldComponent, str] = {
    "re": "real part",
    "im": "imaginary part",
    "abs": "amplitude",
}


def plot_near_field_prepare(
    density: Callable[[Array], Array],
    incident_field: Callable[[Array], Array],
    /,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n: int,
    shapes: Shapes,
    k: Array,
    alpha: Array,
    eta: Array,
    isin_shape_n_quad: int = 500,
    n_plot: int = 100,
    isin_shape_tol: float = 1e-5,
) -> FieldData:
    r"""
    Precompute near-field data on a grid for plotting.

    Parameters
    ----------
    density : Callable[[Array], Array]
        Density function of shape (...) -> (..., M).
    incident_field : Callable[[Array], Array]
        Incident field of shape (..., 2) -> (...,).
    xlim : tuple[float, float]
        Horizontal extent of the plot domain.
    ylim : tuple[float, float]
        Vertical extent of the plot domain.
    n : int
        Maximum order minus $1$.
    shapes : Shapes
        Boundary parametrisations of the $M$ scatterers.
    k : Array
        Wave number.
    alpha : Array
        Coupling parameter of shape (M,).
    eta : Array
        Coupling parameter of shape (M,).
    isin_shape_n_quad : int
        Number of quadrature nodes for the inside-shape test.
    n_plot : int
        Number of grid points per axis.
    isin_shape_tol : float
        Winding-number threshold for the inside-shape test.

    Returns
    -------
    FieldData
        Dictionary keyed by field/component tuples.

    """
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device
    x = xp.linspace(xlim[0], xlim[1], n_plot, device=device, dtype=dtype)
    y = xp.linspace(ylim[0], ylim[1], n_plot, device=device, dtype=dtype)
    x, y = xp.broadcast_arrays(x[:, None], y[None, :])
    xy = xp.stack([x, y], axis=-1)
    uscat = near_field(density, xy, n=n, shapes=shapes, k=k, alpha=alpha, eta=eta)
    uin = incident_field(xy)
    utot = uscat + uin
    inside = isin_shapes(xy, shapes, n_quad=isin_shape_n_quad, tol=isin_shape_tol)
    uscat[inside] = xp.nan
    uin[inside] = xp.nan
    utot[inside] = xp.nan

    extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    result: FieldData = {}
    for field_name, field_val in (("uin", uin), ("uscat", uscat), ("utot", utot)):
        valid_reim = xp.abs(
            xp.concat([
                field_val.real[~xp.isnan(field_val.real)],
                field_val.imag[~xp.isnan(field_val.imag)],
            ])
        )
        vmax_reim = float(quantile(valid_reim, 0.99))
        vmax_abs = float(quantile(xp.abs(field_val[~xp.isnan(field_val)]), 0.99))
        result[field_name] = {}  # type: ignore
        for component in ("re", "im", "abs"):
            if component == "re":
                data = field_val.real
                vmax = vmax_reim
                vmin = -vmax
            elif component == "im":
                data = field_val.imag
                vmax = vmax_reim
                vmin = -vmax
            else:
                data = xp.abs(field_val)
                vmax = vmax_abs
                vmin = 0
            result[field_name][component] = {  # type: ignore
                "data": xp.ascontiguousarray(data),
                "vmax": vmax,
                "vmin": vmin,
                "extent": extent,
            }
    return result


def plot_near_field(
    field_data: FieldData,
    /,
    *,
    ax_uin_re: Axes | None = None,
    ax_uin_im: Axes | None = None,
    ax_uin_abs: Axes | None = None,
    ax_uscat_re: Axes | None = None,
    ax_uscat_im: Axes | None = None,
    ax_uscat_abs: Axes | None = None,
    ax_utot_re: Axes | None = None,
    ax_utot_im: Axes | None = None,
    ax_utot_abs: Axes | None = None,
) -> None:
    r"""
    Plot precomputed near-field data on provided axes.

    Parameters
    ----------
    field_data : FieldData
        Output of :func:`plot_near_field_prepare`.
    ax_uin_re : Axes | None
        Axes for incident field real part.
    ax_uin_im : Axes | None
        Axes for incident field imaginary part.
    ax_uin_abs : Axes | None
        Axes for incident field amplitude.
    ax_uscat_re : Axes | None
        Axes for scattered field real part.
    ax_uscat_im : Axes | None
        Axes for scattered field imaginary part.
    ax_uscat_abs : Axes | None
        Axes for scattered field amplitude.
    ax_utot_re : Axes | None
        Axes for total field real part.
    ax_utot_im : Axes | None
        Axes for total field imaginary part.
    ax_utot_abs : Axes | None
        Axes for total field amplitude.

    """
    axes: list[tuple[tuple[FieldKind, FieldComponent], Axes | None]] = [
        (("uin", "re"), ax_uin_re),
        (("uin", "im"), ax_uin_im),
        (("uin", "abs"), ax_uin_abs),
        (("uscat", "re"), ax_uscat_re),
        (("uscat", "im"), ax_uscat_im),
        (("uscat", "abs"), ax_uscat_abs),
        (("utot", "re"), ax_utot_re),
        (("utot", "im"), ax_utot_im),
        (("utot", "abs"), ax_utot_abs),
    ]
    for key, ax in axes:
        if ax is None:
            continue
        entry = field_data[key[0]][key[1]]
        cmap = "inferno" if key[1] == "abs" else "seismic"
        im = ax.imshow(
            np.asarray(entry["data"], dtype=float).T,
            extent=entry["extent"],
            origin="lower",
            cmap=cmap,
            vmin=entry["vmin"],
            vmax=entry["vmax"],
        )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{_FIELD_NAMES[key[0]]} field {_COMPONENT_NAMES[key[1]]}")


def near_field(
    density: Callable[[Array], Array],
    x: Array,
    /,
    *,
    n: int,
    shapes: Shapes,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Array:
    r"""
    Compute near field for multiple scatterers.

    $$
    u^s = \\sum_{j=1}^M (\\alpha_j D_j - i \\eta_j S_j) \\phi_j
    $$

    Parameters
    ----------
    density : Callable[[Array], Array]
        The density function of shape (...) -> (..., M).
    x : Array
        The position of the scattered field of shape (..., 2).
    n : int
        The maximum order - 1.
    shapes : Shapes
        The shapes of the $M$ scatterers.
    k : Array
        The wave number of shape (...(B),).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (M,).
    eta : Array
        The coupling parameter for the single-layer potential of shape (M,).

    Returns
    -------
    Array
        The near field of shape (..., ...(*B)).

    """
    M = shapes.n_shapes
    result = None
    for j in range(M):
        shape_j = shapes[j]

        def density_j(t: Array, _j: int = j) -> Array:
            return density(t)[..., _j]

        dlp_ = dlp(x, density_j, shape_x=shape_j.x, shape_dx=shape_j.dx, k=k, n=n)
        slp_ = slp(x, density_j, shape_x=shape_j.x, shape_dx=shape_j.dx, k=k, n=n)
        contrib = alpha[j] * dlp_ - 1j * eta[j] * slp_
        result = contrib if result is None else result + contrib
    return result


def far_field(
    density: Callable[[Array], Array],
    direction: Array,
    /,
    *,
    n: int,
    shapes: Shapes,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Array:
    """
    Compute far-field pattern for multiple scatterers.

    Parameters
    ----------
    density : Callable[[Array], Array]
        The density function of shape (...) -> (..., M).
    direction : Array
        The direction of the far-field pattern of shape (..., 2).
    n : int
        The maximum order - 1.
    shapes : Shapes
        The shapes of the $M$ scatterers.
    k : Array
        The wave number of shape (...(B),).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (M,).
    eta : Array
        The coupling parameter for the single-layer potential of shape (M,).

    Returns
    -------
    Array
        The far-field pattern of shape (..., ...(*B)).

    """
    xp = array_namespace(direction, k)
    dtype = xp.result_type(direction, k)
    device = direction.device
    t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    coef = xp.exp(-1j * xp.pi / 4) / xp.sqrt(8 * xp.pi * k)
    direction_normalized = direction / xp.linalg.vector_norm(direction, axis=-1, keepdims=True)

    # Batched computation over all shapes
    # shapes.x(t) -> (Q, M, 2), shapes.dx(t) -> (Q, M, 2)
    x_t = shapes.x(t)  # (Q, M, 2)
    dx_t = shapes.dx(t)  # (Q, M, 2)
    ny_t_unnormalized = xp.stack([dx_t[..., 1], -dx_t[..., 0]], axis=-1)  # (Q, M, 2)
    ny_t = ny_t_unnormalized / xp.linalg.vector_norm(
        ny_t_unnormalized, axis=-1, keepdims=True
    )  # (Q, M, 2)
    jacobian = xp.sqrt(xp.sum(dx_t**2, axis=-1))  # (Q, M)

    # (Q, M)
    integrand_without_density = (
        (alpha * k * xp.sum(ny_t * direction_normalized[..., None, :], axis=-1) + eta)
        * xp.exp(-1j * k * xp.sum(x_t * direction[..., None, :], axis=-1))
        * jacobian
    )

    # density(t) -> (Q, M)
    density_t = density(t)  # (Q, M)
    integrand = integrand_without_density * density_t  # (Q, M)

    # Sum over quadrature points for each shape, then sum over shapes
    integral_per_shape = xp.sum(integrand * w[..., None], axis=0)  # (M,)
    integral = xp.sum(integral_per_shape, axis=-1)  # scalar

    return coef * integral
