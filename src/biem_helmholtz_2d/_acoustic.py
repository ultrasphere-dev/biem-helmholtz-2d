import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import Shape, nystrom, trapezoidal_quadrature
from ie_circle._bie import Kernels, NystromInterpolant, QuadratureType
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import quantile

from ._potential import dlp_kernel_split, slp_kernel_split
from ._potential_inner import dlp_kernel, slp_kernel


def _coupled_kernels(*, shape: Sequence[Shape], k: Array, alpha: Array, eta: Array) -> Kernels:
    xp = array_namespace(k, alpha, eta)
    k, alpha, eta = xp.broadcast_arrays(k, alpha, eta)
    info = check_shapes("*B,*B,*B", k, alpha, eta, names="k,alpha,eta")
    batch_ndim = len(info.unique["B"].shape_broadcasted)

    def coupled_kernel(t: Array, tau: Array, /, *, logarithmic: bool) -> Array:
        check_shapes("...,...", t, tau, names="t,tau")
        t = t[(...,) + (None,) * batch_ndim]
        tau = tau[(...,) + (None,) * batch_ndim]
        rows = []
        for i, target in enumerate(shape):
            columns = []
            for j, source in enumerate(shape):
                if i == j:
                    s = slp_kernel_split(t=t, tau=tau, k=k, x=source.x, dx=source.dx)
                    d = dlp_kernel_split(
                        t=t, tau=tau, k=k, x=source.x, dx=source.dx, ddx=source.ddx
                    )
                    part = 0 if logarithmic else 1
                    value = alpha * d[part] - 1j * eta * s[part]
                elif logarithmic:
                    value = xp.zeros_like(t + tau + 0 * k, dtype=xp.result_type(k, alpha, eta, 1j))
                else:
                    point = target.x(t)
                    value = alpha * dlp_kernel(
                        point, shape_x=source.x, shape_dx=source.dx, k=k, tau=tau
                    ) - 1j * eta * slp_kernel(
                        point, shape_x=source.x, shape_dx=source.dx, k=k, tau=tau
                    )
                columns.append(value)
            rows.append(xp.stack(columns, axis=-1))
        return xp.stack(rows, axis=-2)

    return {
        (QuadratureType.NO_SINGULARITY, 0): lambda t, tau: coupled_kernel(
            t, tau, logarithmic=False
        ),
        (QuadratureType.LOG_COT_POWER, 0): lambda t, tau: coupled_kernel(t, tau, logarithmic=True),
    }


def scattering_dirichlet(
    *,
    k: Array,
    shape: Sequence[Shape],
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
    shape : Sequence[Shape]
        Disjoint boundaries of (...,) -> (..., 2). The solve is coupled.
    incident_field : Callable[[Array], Array]
        The incident field of (..., 2) -> (..., ...(B)).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (...(B),).
    eta : Array
        The coupling parameter for the single-layer potential of shape (...(B),).
    n : int
        The maximum order - 1.

    Returns
    -------
    NystromInterpolant
        Density of (...) -> (..., ...(B), C), where C is the number of boundaries.

    """
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta))
    device = k.device

    if not shape:
        raise ValueError("At least one boundary is required")
    k, alpha, eta = xp.broadcast_arrays(k, alpha, eta)
    info = check_shapes("*B,*B,*B", k, alpha, eta, names="k,alpha,eta")
    batch_shape = info.unique["B"].shape_broadcasted

    def a(t: Array) -> Array:
        coefficient = xp.broadcast_to(alpha / 2, (*t.shape, *batch_shape))
        return coefficient[..., None] * xp.ones(len(shape), dtype=t.dtype, device=t.device)

    def rhs(t: Array) -> Array:
        parameter = t[(...,) + (None,) * len(batch_shape)]
        components = []
        for boundary in shape:
            values = incident_field(boundary.x(parameter))
            check_shapes("...*B,...,*B", values, t, alpha, names="incident_field,t,alpha")
            if values.shape[: t.ndim] != t.shape:
                raise ValueError("incident_field must have one value per parameter")
            components.append(xp.broadcast_to(-values, (*t.shape, *batch_shape)))
        return xp.stack(components, axis=-1)

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
    xp = array_namespace(x)
    t, w = trapezoidal_quadrature(n_quad, xp=xp, device=x.device, dtype=x.dtype)
    x_t = shape.x(t)
    nx_t_unnormalized = xp.stack([shape.dx(t)[..., 1], -shape.dx(t)[..., 0]], axis=-1)
    diff = x_t - x[..., None, :]
    lower = xp.sum(diff**2, axis=-1)
    upper = xp.sum(nx_t_unnormalized * diff, axis=-1)
    integrand = upper / lower
    integral = xp.sum(integrand * w, axis=-1)
    winding_number = integral / (2 * math.pi)
    return xp.abs(winding_number) > tol


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
    shape: Sequence[Shape],
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
        Maximum order minus $1$.
    shape : Sequence[Shape]
        Boundary parametrisations of (...,) -> (..., 2).
    k : Array
        Wave number of shape (...(B),).
    alpha : Array
        Coupling parameter of shape (...(B),).
    eta : Array
        Coupling parameter of shape (...(B),).
    isin_shape_n_quad : int
        Number of quadrature nodes for the inside-shape test.
    n_plot : int
        Number of grid points per axis.
    isin_shape_tol : float
        Winding-number threshold for the inside-shape test.

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
    uscat = near_field(density, xy, n=n, shape=shape, k=k, alpha=alpha, eta=eta)
    batch_ndim = uscat.ndim - (xy.ndim - 1)
    uin = incident_field(xy[(...,) + (None,) * batch_ndim + (slice(None),)])
    uin = xp.broadcast_to(uin, uscat.shape)
    utot = uscat + uin
    inside = xp.any(
        xp.stack([
            isin_shape(xy, boundary, n_quad=isin_shape_n_quad, tol=isin_shape_tol)
            for boundary in shape
        ]),
        axis=0,
    )
    inside = inside[(...,) + (None,) * batch_ndim]
    uscat = xp.where(inside, xp.nan, uscat)
    uin = xp.where(inside, xp.nan, uin)
    utot = xp.where(inside, xp.nan, utot)

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
        cmap = "inferno" if key[1] == "abs" else "seismic"
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


def _density_component(
    density: Callable[[Array], Array], index: int, /
) -> Callable[[Array], Array]:
    def component(t: Array, /) -> Array:
        return density(t)[..., index]

    return component


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
    shape: Sequence[Shape],
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
        The position of the scttered field of shape (..., 2).
    n : int
        The maximum order - 1.
    shape : Sequence[Shape]
        Boundaries of (...,) -> (..., 2), in density-component order.
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
    if not shape:
        raise ValueError("At least one boundary is required")
    xp = array_namespace(x, k, alpha, eta)
    point, t, w, k, alpha, eta, values = _field_quadrature(
        density, x, n=n, n_scatterers=len(shape), k=k, alpha=alpha, eta=eta
    )
    fields = []
    for i, boundary in enumerate(shape):
        dlp_ = dlp_kernel(point, shape_x=boundary.x, shape_dx=boundary.dx, k=k, tau=t)
        slp_ = slp_kernel(point, shape_x=boundary.x, shape_dx=boundary.dx, k=k, tau=t)
        fields.append(xp.sum((alpha * dlp_ - 1j * eta * slp_) * values[..., i] * w, axis=-1))
    return xp.sum(xp.stack(fields), axis=0)


def far_field(
    density: Callable[[Array], Array],
    direction: Array,
    /,
    *,
    n: int,
    shape: Sequence[Shape],
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
        The maximum order - 1.
    shape : Sequence[Shape]
        Boundaries of (...,) -> (..., 2), in density-component order.
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
    if not shape:
        raise ValueError("At least one boundary is required")
    xp = array_namespace(direction, k, alpha, eta)
    direction, t, w, k, alpha, eta, values = _field_quadrature(
        density, direction, n=n, n_scatterers=len(shape), k=k, alpha=alpha, eta=eta
    )
    coef = xp.exp(-1j * xp.pi / 4) / xp.sqrt(8 * xp.pi * k)
    direction_normalized = direction / xp.linalg.vector_norm(direction, axis=-1, keepdims=True)
    fields = []
    for i, boundary in enumerate(shape):
        x_t = boundary.x(t)
        dx_t = boundary.dx(t)
        normal = xp.stack([dx_t[..., 1], -dx_t[..., 0]], axis=-1)
        jacobian = xp.linalg.vector_norm(normal, axis=-1)
        normal = normal / jacobian[..., None]
        kernel = (
            (alpha * k * xp.sum(normal * direction_normalized, axis=-1) + eta)
            * xp.exp(-1j * k * xp.sum(x_t * direction, axis=-1))
            * jacobian
        )
        fields.append(xp.sum(coef * kernel * values[..., i] * w, axis=-1))
    return xp.sum(xp.stack(fields), axis=0)
