from collections.abc import Callable

from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import Shape, nystrom, trapezoidal_quadrature
from ie_circle._bie import NystromInterpolant, QuadratureType
from matplotlib import pyplot as plt

from ._potential import dlp_kernel_split, slp_kernel_split


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
        The wave number of shape (...,).
    shape : Shape
        The shape of the scatterer of (...,) -> (..., 2).
    incident_field : Callable[[Array], Array]
        The incident field of (...,) -> (..., 2).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (...,).
    eta : Array
        The coupling parameter for the single-layer potential of shape (...,).
    n : int
        The maximum order - 1.

    Returns
    -------
    NystromInterpolant
        The density of the integral equation of shape (...,) -> (..., 1, 1).

    """
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device

    def k_log(t: Array, tau: Array) -> Array:
        slp_log, _ = slp_kernel_split(t, tau, k[..., None, None], shape.x, shape.dx)
        dlp_log, _ = dlp_kernel_split(t, tau, k[..., None, None], shape.x, shape.dx, shape.ddx)
        res = alpha * dlp_log - 1j * eta * slp_log
        res = res[..., None, None]
        return res

    def k_cont(t: Array, tau: Array) -> Array:
        _, slp_cont = slp_kernel_split(t, tau, k[..., None, None], shape.x, shape.dx)
        _, dlp_cont = dlp_kernel_split(t, tau, k[..., None, None], shape.x, shape.dx, shape.ddx)
        res = alpha * dlp_cont - 1j * eta * slp_cont
        res = res[..., None, None]
        return res

    def a(t: Array) -> Array:
        xp = array_namespace(t)
        return xp.ones_like(t)[..., None] * (alpha / 2)

    def rhs(t: Array) -> Array:
        return -incident_field(shape.x(t))[..., None]

    kernels = {
        (QuadratureType.NO_SINGULARITY, 0): k_cont,
        (QuadratureType.LOG_COT_POWER, 0): k_log,
    }

    result = nystrom(a, kernels, rhs, n=n, xp=xp, device=device, dtype=dtype)
    return result


from ._potential_inner import dlp, slp


def plot_ner_field(
    density: Callable[[Array], Array],
    /,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n: int,
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
) -> None:
    xp = array_namespace(k, alpha, eta)
    dtype = xp.result_type(k, alpha, eta)
    device = k.device
    x = xp.linspace(xlim[0], xlim[1], 100, device=device, dtype=dtype)
    y = xp.linspace(ylim[0], ylim[1], 100, device=device, dtype=dtype)
    x, y = xp.broadcast_arrays(x[:, None], y[None, :])
    xy = xp.stack([x, y], axis=-1)
    u = near_field(density, xy, n=n, shape=shape, k=k, alpha=alpha, eta=eta)
    im = plt.imshow(xp.abs(u).T, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin="lower")
    plt.colorbar(im)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Near Field (Absolute Value)")


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
        The density function of shape (...) -> (..., ...(B), 1, 1).
    x : Array
        The position of the scttered field of shape (..., 2).
    n : int
        The maximum order - 1.
    shape : Shape
        The shape of the scatterer of (...,) -> (..., 2).
    k : Array
        The wave number of shape (...(B),).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (...(B),).
    eta : Array
        The coupling parameter for the single-layer potential of shape (...(B),).

    Returns
    -------
    Array
        The near field of shape (..., ...(*B)).

    """
    dlp_ = dlp(x, density, shape_x=shape.x, shape_dx=shape.dx, k=k, n=n)
    slp_ = slp(x, density, shape_x=shape.x, shape_dx=shape.dx, k=k, n=n)
    return alpha * dlp_ - 1j * eta * slp_


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
        The density function of shape (...) -> (..., ...(B), 1, 1).
    direction : Array
        The direction of the far-field pattern of shape (..., 2).
    n : int
        The maximum order - 1.
    shape : Shape
        The shape of the scatterer of (...,) -> (..., 2).
    k : Array
        The wave number of shape (...(B),).
    alpha : Array
        The coupling parameter for the double-layer potential of shape (...(B),).
    eta : Array
        The coupling parameter for the single-layer potential of shape (...(B),).

    Returns
    -------
    Array
        The far-field pattern of shape (..., ...(*B)).

    """
    xp = array_namespace(direction, k)
    dtype = xp.result_type(direction, k)
    device = direction.device
    t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    # (...)
    coef = xp.exp(-1j * xp.pi / 4) / xp.sqrt(8 * xp.pi * k)
    x_t = shape.x(t)
    dx_t = shape.dx(t)
    ny_t_unnormalized = xp.stack([dx_t[..., 1], -dx_t[..., 0]], axis=-1)
    ny_t = ny_t_unnormalized / xp.linalg.vector_norm(ny_t_unnormalized, axis=-1, keepdims=True)
    jacobian = xp.sqrt(xp.sum(dx_t**2, axis=-1))
    direction_normalized = direction / xp.linalg.vector_norm(direction, axis=-1, keepdims=True)
    # (..., Q)
    integrand_without_density = (
        (
            alpha[..., None, None]
            * k[..., None, None]
            * xp.sum(ny_t * direction_normalized, axis=-1)
            + eta[..., None, None]
        )
        * xp.exp(-1j * k[..., None, None] * xp.sum(x_t * direction, axis=-1))
        * jacobian
    )
    # (..., *B, Q)
    density_t = density(t)
    info = check_shapes("*bQ,*b*BQ", density_t, integrand_without_density)
    B_ndim = len(info.unique["b"].shape_broadcasted)
    integrand = (
        integrand_without_density[(...,) + (None,) * B_ndim + (slice(None), slice(None))]
        * density_t
    )
    integral = xp.sum(integrand * w, axis=-1)
    result = coef[..., None] * integral
    return result
