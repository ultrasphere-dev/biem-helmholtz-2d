from collections.abc import Callable

from array_api.latest import Array
from array_api_compat import array_namespace
from ie_circle import nystrom
from ie_circle._bie import QuadratureType

from ._potential import dlp, slp
from ._shape import Shape


def scattering_dirichlet(
    *,
    k: complex,
    shape: Shape,
    incident_field: Callable[[Array], Array],
    alpha: complex,
    eta: complex,
    n: int,
) -> Array:
    """
    Compute scattering field.

    Parameters
    ----------
    k : complex
        The wave number.
    shape : Shape
        The shape of the scatterer.
    incident_field : Callable[[Array], Array]
        The incident field of (...,) -> (..., 2).
    alpha : complex
        The coupling parameter for the double-layer potential.
    eta : complex
        The coupling parameter for the single-layer potential.
    n : int
        The maximum order - 1.

    Returns
    -------
    Array
        _description_

    """

    def k_log(t: Array, tau: Array) -> Array:
        slp_log, _ = slp(t, tau, k, shape.x, shape.dx)
        dlp_log, _ = dlp(t, tau, k, shape.x, shape.dx, shape.ddx)
        res = alpha * slp_log + eta * dlp_log
        return res[..., None, None]

    def k_cont(t: Array, tau: Array) -> Array:
        _, slp_rem = slp(t, tau, k, shape.x, shape.dx)
        _, dlp_rem = dlp(t, tau, k, shape.x, shape.dx, shape.ddx)
        res = alpha * slp_rem + eta * dlp_rem
        return res[..., None, None]

    def a(t: Array) -> Array:
        xp = array_namespace(t)
        return xp.ones_like(t)[..., None, None] * (alpha / 2)

    def rhs(t: Array) -> Array:
        return incident_field(shape.x(t))[..., None]

    kernels = {
        (QuadratureType.NO_SINGULARITY, 0): k_cont,
        (QuadratureType.LOG_COT_POWER, 0): k_log,
    }
    return nystrom(a, kernels, rhs, n=n)
