import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import Shape, nystrom, trapezoidal_quadrature
from ie_circle._bie import Kernels, NystromInterpolant, QuadratureType
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ._pairs import off_diagonal_indices, off_diagonal_matrix
from ._potential import dlp_kernel_split, slp_kernel_split
from ._potential_far import kernel as far_kernel
from ._potential_inner import dlp_kernel, slp_kernel
from ._potential_inner import kernel as near_kernel


def _coupled_kernels(*, shape: Shape, k: Array, alpha: Array, eta: Array) -> Kernels:
    xp = array_namespace(k, alpha, eta)
    k, alpha, eta = xp.broadcast_arrays(k, alpha, eta)
    batch_ndim = k.ndim

    def coupled_kernel(t: Array, tau: Array, /, *, logarithmic: bool) -> Array:
        check_shapes("...,...", t, tau, names="t,tau")
        t = t[(...,) + (None,) * batch_ndim]
        tau = tau[(...,) + (None,) * batch_ndim]
        x_t, x_tau = shape.x(t), shape.x(tau)
        dx_tau, ddx_tau = shape.dx(tau), shape.ddx(tau)
        count = x_t.shape[-2]
        t_diag, tau_diag = t[..., None], tau[..., None]
        s = slp_kernel_split(
            t=t_diag, tau=tau_diag, k=k[..., None], x_t=x_t, x_tau=x_tau, dx_tau=dx_tau
        )
        d = dlp_kernel_split(
            t=t_diag,
            tau=tau_diag,
            k=k[..., None],
            x_t=x_t,
            x_tau=x_tau,
            dx_tau=dx_tau,
            ddx_tau=ddx_tau,
        )
        part = 0 if logarithmic else 1
        diagonal = alpha[..., None] * d[part] - 1j * eta[..., None] * s[part]
        matrix = diagonal[..., :, None] * xp.eye(count, dtype=diagonal.dtype, device=k.device)
        if not logarithmic and count > 1:
            rows, cols = off_diagonal_indices(count, xp=xp, device=k.device)
            point = xp.take(x_t, rows, axis=-2)
            source = xp.take(x_tau, cols, axis=-2)
            derivative = xp.take(dx_tau, cols, axis=-2)
            value = alpha[..., None] * dlp_kernel(
                point, shape_x=source, shape_dx=derivative, k=k[..., None]
            ) - 1j * eta[..., None] * slp_kernel(
                point, shape_x=source, shape_dx=derivative, k=k[..., None]
            )
            matrix = matrix + off_diagonal_matrix(value, count)
        return matrix

    return {
        (QuadratureType.NO_SINGULARITY, 0): lambda t, tau: coupled_kernel(
            t, tau, logarithmic=False
        ),
        (QuadratureType.LOG_COT_POWER, 0): lambda t, tau: coupled_kernel(t, tau, logarithmic=True),
    }


def scattering_dirichlet(
    *,
    k: Array,
    shape: Shape,
    incident_field: Callable[[Array], Array],
    alpha: Array,
    eta: Array,
    n: int,
) -> NystromInterpolant:
    """
    Compute scattering field.

    Parameters
    ----------
    k : Array
        The wave number of shape (...(B),).
    shape : Shape
        Disjoint boundaries of (...,) -> (..., C, 2). The solve is coupled.
    incident_field : Callable[[Array], Array]
        The incident field of (..., 2) -> (..., ...(B)).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (...(B),).
    eta : Array
        The coupling parameter for the single-layer potential of shape (...(B),).
    n : int
        Nyström parameter; the solve uses $2n-1$ nodes.

    Returns
    -------
    NystromInterpolant
        Density of (...) -> (..., ...(B), C), where C is the number of boundaries.

    """
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta))
    device = k.device

    k, alpha, eta = xp.broadcast_arrays(k, alpha, eta)
    batch_shape = k.shape
    count = shape.x(xp.asarray(0, dtype=dtype, device=device)).shape[-2]
    if count == 0:
        raise ValueError("At least one boundary is required")

    def a(t: Array) -> Array:
        coefficient = xp.broadcast_to(alpha / 2, (*t.shape, *batch_shape))
        return coefficient[..., None] * xp.ones(count, dtype=t.dtype, device=t.device)

    def rhs(t: Array) -> Array:
        parameter = t[(...,) + (None,) * len(batch_shape)]
        points = xp.moveaxis(shape.x(parameter), -2, 0)
        values = incident_field(points)
        check_shapes(
            "C...*B,...,*B,C", values, t, alpha, (count,), names="incident_field,t,alpha,C"
        )
        if values.shape[0] != count or values.shape[1 : t.ndim + 1] != t.shape:
            raise ValueError("incident_field must have one value per parameter and scatterer")
        return xp.broadcast_to(-xp.moveaxis(values, 0, -1), (*t.shape, *batch_shape, count))

    return nystrom(
        a,
        _coupled_kernels(shape=shape, k=k, alpha=alpha, eta=eta),
        rhs,
        n=n,
        xp=xp,
        device=device,
        dtype=dtype,
    )


def isin_shape(x: Array, shape: Shape, /, n_quad: int, tol: float = 1e-5) -> Array:
    """Classify points using the winding number of the sampled boundary polygon."""
    xp = array_namespace(x)
    t, _ = trapezoidal_quadrature(n_quad, xp=xp, device=x.device, dtype=x.dtype)
    boundary = shape.x(t)
    start = boundary - x[..., None, None, :]
    end = xp.roll(boundary, -1, axis=0) - x[..., None, None, :]
    cross = start[..., 0] * end[..., 1] - start[..., 1] * end[..., 0]
    dot = xp.sum(start * end, axis=-1)
    winding_number = xp.sum(xp.atan2(cross, dot), axis=-2) / (2 * math.pi)
    return xp.abs(winding_number) > max(0.5, tol)


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
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
    isin_shape_n_quad: int = 500,
    n_plot: int = 100,
    isin_shape_tol: float = 1e-5,
) -> FieldData:
    r"""
    Precompute near-field data on a grid for plotting.

    Computes the incident, scattered, and total fields on a regular grid,
    masks points inside the scatterer with NaN, and determines
    per-field color limits.

    Parameters
    ----------
    density : Callable[[Array], Array]
        Density function of shape (...) -> (..., ...(B), C), where C is the number of boundaries.
    incident_field : Callable[[Array], Array]
        Incident field of shape (..., 2) -> (..., ...(B)).
    xlim : tuple[float, float]
        Horizontal extent of the plot domain.
    ylim : tuple[float, float]
        Vertical extent of the plot domain.
    n : int
        Integration parameter; the field uses $2n-1$ nodes.
    shape : Shape
        Boundary parametrisations of (...) -> (..., C, 2).
    k : Array
        Wave number of shape (...(B),).
    alpha : Array
        Coupling parameter of shape (...(B),).
    eta : Array
        Coupling parameter of shape (...(B),).
    isin_shape_n_quad : int
        Quadrature parameter for the inside-shape test; uses $2n-1$ nodes.
    n_plot : int
        Number of grid points per axis.
    isin_shape_tol : float
        Winding-number threshold for the inside-shape test, with a minimum of 0.5.

    Returns
    -------
    FieldData
        Dictionary keyed by field (``"uin"``, ``"uscat"``, ``"utot"``),
        then component (``"re"``, ``"im"``, ``"abs"``). Each entry has
        keys ``"data"``, ``"vmax"``, ``"vmin"``, and ``"extent"``.
        The data arrays have shape ``(n_plot, n_plot, ...(B))``.

    """
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta))
    device = k.device
    x = xp.linspace(xlim[0], xlim[1], n_plot, device=device, dtype=dtype)
    y = xp.linspace(ylim[0], ylim[1], n_plot, device=device, dtype=dtype)
    x, y = xp.broadcast_arrays(x[:, None], y[None, :])
    xy = xp.stack([x, y], axis=-1)
    uscat = xp.concat(
        [
            near_field(density, xy[i : i + 8], n=n, shape=shape, k=k, alpha=alpha, eta=eta)
            for i in range(0, n_plot, 8)
        ],
        axis=0,
    )
    batch_ndim = uscat.ndim - (xy.ndim - 1)
    uin = incident_field(xy[(...,) + (None,) * batch_ndim + (slice(None),)])
    uin = xp.broadcast_to(uin, uscat.shape)
    utot = uscat + uin
    inside = xp.concat(
        [
            xp.any(
                isin_shape(xy[i : i + 8], shape, n_quad=isin_shape_n_quad, tol=isin_shape_tol),
                axis=-1,
            )
            for i in range(0, n_plot, 8)
        ],
        axis=0,
    )
    inside = inside[(...,) + (None,) * batch_ndim]
    uscat = xp.where(inside, xp.nan, uscat)
    uin = xp.where(inside, xp.nan, uin)
    utot = xp.where(inside, xp.nan, utot)

    extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    def quantile_99(values: Array, /) -> float:
        ordered = xp.sort(values)
        position = 0.99 * (ordered.shape[0] - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        return float(ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]))

    result: FieldData = {}
    for field_name, field_val in (("uin", uin), ("uscat", uscat), ("utot", utot)):
        valid_reim = xp.abs(
            xp.concat([
                field_val.real[~xp.isnan(field_val.real)],
                field_val.imag[~xp.isnan(field_val.imag)],
            ])
        )
        vmax_reim = quantile_99(valid_reim)
        vmax_abs = quantile_99(xp.abs(field_val[~xp.isnan(field_val)]))
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
                "data": data,
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

    Select a single batch from the data arrays before plotting batched fields.

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
        data = entry["data"]
        if isinstance(data, (list, tuple)):
            data = [
                [float("nan") if value is None else value for value in column]
                for column in zip(*data, strict=True)
            ]
        else:
            check_shapes("IJ", data, names="field_data")
            xp = array_namespace(data)
            data = xp.permute_dims(xp.asarray(data, device="cpu"), (1, 0))
        cmap = plt.get_cmap("inferno" if key[1] == "abs" else "seismic").with_extremes(bad="gray")
        im = ax.imshow(
            data,
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


def _field_quadrature(
    density: Callable[[Array], Array],
    points: Array,
    /,
    *,
    n: int,
    n_scatterers: int,
    k: Array,
    alpha: Array,
    eta: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    check_shapes("...D,D", points, (2,), names="points,coordinate_dimension")
    if points.shape[-1] != 2:
        raise ValueError("points must contain two coordinates")
    xp = array_namespace(points, k, alpha, eta)
    t, w = trapezoidal_quadrature(
        n, xp=xp, device=points.device, dtype=xp.result_type(points, xp.real(k))
    )
    values = density(t)
    info = check_shapes("Q*BC,QC", values, (t.shape[0], n_scatterers), names="density,expected")
    if values.shape[0] != t.shape[0] or values.shape[-1] != n_scatterers:
        raise ValueError("density must have one value per quadrature node and scatterer")
    density_batch_ndim = len(info.unique["B"].shape_broadcasted)
    k, alpha, eta, _ = xp.broadcast_arrays(k, alpha, eta, values[0, ..., 0])
    batch_shape = k.shape
    batch_ndim = len(batch_shape)
    values = values[(slice(None),) + (None,) * (batch_ndim - density_batch_ndim) + (...,)]
    check_shapes(
        "Q*BC,*B,*B,*B,QC",
        values,
        k,
        alpha,
        eta,
        (t.shape[0], n_scatterers),
        names="density,k,alpha,eta,expected",
    )
    values = xp.moveaxis(xp.broadcast_to(values, (t.shape[0], *batch_shape, n_scatterers)), 0, -2)
    point_ndim = points.ndim - 1
    points = points[(...,) + (None,) * (batch_ndim + 1) + (slice(None),)]
    t = t[(None,) * (point_ndim + batch_ndim) + (slice(None),)]
    expand = (None,) * point_ndim + (..., None)
    return points, t, w, k[expand], alpha[expand], eta[expand], values


def near_field(
    density: Callable[[Array], Array],
    x: Array,
    /,
    *,
    n: int,
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Array:
    r"""
    Compute near field.

    $$
    u = (\alpha D - i \eta S) \phi
    $$

    Parameters
    ----------
    density : Callable[[Array], Array]
        The density function of shape (...) -> (..., ...(B), C),
        where C is the number of boundaries.
    x : Array
        The position of the scattered field of shape (..., 2).
    n : int
        Integration parameter, with $2n-1$ nodes. It may differ from the
        Nyström parameter used to compute ``density``.
    shape : Shape
        Boundaries of (...) -> (..., C, 2), in density-component order.
    k : Array
        The wave number of shape (...(B),).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (...(B),).
    eta : Array
        The coupling parameter for the single-layer potential of shape (...(B),).

    Returns
    -------
    Array
        The near field of shape (..., ...(B)).

    """
    xp = array_namespace(x, k, alpha, eta)
    count = shape.x(xp.asarray(0, dtype=x.dtype, device=x.device)).shape[-2]
    point, t, w, k, alpha, eta, values = _field_quadrature(
        density, x, n=n, n_scatterers=count, k=k, alpha=alpha, eta=eta
    )
    x_t, dx_t = shape.x(t), shape.dx(t)
    kernel = near_kernel(
        point[..., None, :],
        shape_x=x_t,
        shape_dx=dx_t,
        k=k[..., None],
        alpha=alpha[..., None],
        eta=eta[..., None],
    )
    return xp.sum(kernel * values * w[..., None], axis=(-2, -1))


def far_field(
    density: Callable[[Array], Array],
    direction: Array,
    /,
    *,
    n: int,
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Array:
    """
    Compute far-field pattern.

    Parameters
    ----------
    density : Callable[[Array], Array]
        The density function of shape (...) -> (..., ...(B), C),
        where C is the number of boundaries.
    direction : Array
        The direction of the far-field pattern of shape (..., 2).
    n : int
        Integration parameter, with $2n-1$ nodes. It may differ from the
        Nyström parameter used to compute ``density``.
    shape : Shape
        Boundaries of (...) -> (..., C, 2), in density-component order.
    k : Array
        The wave number of shape (...(B),).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (...(B),).
    eta : Array
        The coupling parameter for the single-layer potential of shape (...(B),).

    Returns
    -------
    Array
        The far-field pattern of shape (..., ...(B)).

    """
    xp = array_namespace(direction, k, alpha, eta)
    count = shape.x(xp.asarray(0, dtype=direction.dtype, device=direction.device)).shape[-2]
    direction, t, w, k, alpha, eta, values = _field_quadrature(
        density, direction, n=n, n_scatterers=count, k=k, alpha=alpha, eta=eta
    )
    x_t, dx_t = shape.x(t), shape.dx(t)
    kernel = far_kernel(
        direction[..., None, :],
        shape_x=x_t,
        shape_dx=dx_t,
        k=k[..., None],
        alpha=alpha[..., None],
        eta=eta[..., None],
    )
    return xp.sum(kernel * values * w[..., None], axis=(-2, -1))
