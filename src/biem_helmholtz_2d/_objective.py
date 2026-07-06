from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import Shape

from ._scipy_wrapper import scipy_hankel1


def grad_phi_scattered_field(
    point: Array,
    field_value: Array,
    /,
    *,
    shape: Shape,
    k: Array,
    alpha: Array,
    eta: Array,
) -> Callable[[Array], Array]:
    r"""
    Riesz representation of $d_\phi j$ for $j = |u|^2$.

    For the objective $j(\phi) = |u(x_0)|^2$ with
    $u = (\alpha\mathcal D - i\eta\mathcal S)\phi$, the Frechet derivative
    with respect to $\phi$ is

    $$
    D_\phi j[\nu] = 2\operatorname{Re}\bigl(
        \overline{u(x_0)}\,
        (\alpha\mathcal D - i\eta\mathcal S)\nu(x_0)
    \bigr).
    $$

    For the $L^2$ bilinear form $\langle f,g\rangle = \int f g$, the Riesz
    representation is

    $$
    \operatorname{grad}_\phi j(\tau)
    = 2\operatorname{Re}\bigl(
        \overline{u(x_0)}\,
        \bigl(\alpha\widetilde{\mathcal D}(x_0,\tau)
              - i\eta\,\widetilde{\mathcal S}(x_0,\tau)\bigr)
    \bigr),
    $$

    where $\widetilde{\mathcal D}(x_0,\tau),\,
          \widetilde{\mathcal S}(x_0,\tau)$ are the parametrised evaluation
    kernels of the double- and single-layer potentials at $x_0$.

    Parameters
    ----------
    point : Array
        Evaluation point $x_0$ of shape (..., 2).
    field_value : Array
        Scattered field $u(x_0)$ of shape (...,).
    shape : Shape
        Boundary parametrisation of (...,) -> (..., 2).
    k : Array
        Wave number of shape (...,).
    alpha : Array
        Coupling parameter $\alpha$ of shape (...,).
    eta : Array
        Coupling parameter $\eta$ of shape (...,).

    Returns
    -------
    Callable[[Array], Array]
        Function $\operatorname{grad}_\phi j$ that takes boundary parameter
        $t$ of shape ``(...,)`` and returns values of shape ``(...,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from ie_circle import CircleShape
    >>> from biem_helmholtz_2d._acoustic import scattering_dirichlet, near_field
    >>> xp = np; k = xp.asarray(1.0); rho = 1.0
    >>> shape = CircleShape(rho)
    >>> a = xp.asarray(1.0); e = xp.asarray(0.0)
    >>> def inc(x): return xp.exp(1j * k * x[..., 0])
    >>> phi = scattering_dirichlet(k=k, shape=shape, incident_field=inc, alpha=a, eta=e, n=8)
    >>> x0 = xp.asarray([3.0, 3.0])
    >>> u = near_field(phi, x0[None], k=k, shape=shape, n=8, alpha=a, eta=e)
    >>> gj = grad_phi_scattered_field(x0[None], u, shape=shape, k=k, alpha=a, eta=e)
    >>> gj(xp.asarray([0.0]))
    array([-0.03860757])

    """
    xp = array_namespace(point, k, alpha, eta)
    u_sq = xp.squeeze(field_value)

    def grad(t_in: Array) -> Array:
        x_t = shape.x(t_in)
        dx_t = shape.dx(t_in)
        diff = point - x_t
        dist = xp.linalg.vector_norm(diff, axis=-1)
        jac = xp.linalg.vector_norm(dx_t, axis=-1)
        h0 = scipy_hankel1(0, k * dist)
        h1 = scipy_hankel1(1, k * dist)
        ny = xp.stack([dx_t[..., 1], -dx_t[..., 0]], axis=-1)
        ny_dot = xp.sum(ny * diff, axis=-1)
        kernel = alpha * ((1j * k / 4) * (ny_dot / dist) * h1) - 1j * eta * ((1j / 4) * h0 * jac)
        return 2.0 * xp.real(xp.conj(u_sq) * kernel)

    return grad
