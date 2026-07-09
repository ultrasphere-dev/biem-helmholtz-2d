from __future__ import annotations

from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ie_circle import Shape

from ._potential_inner import dlp_kernel, slp_kernel


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
    Riesz representation of $d_\phi j$ under the $L^2$ sesquilinear form.

    For the objective $j(\phi) = |u(x_0)|^2$ with
    $u = (\alpha\mathcal D - i\eta\mathcal S)\phi$, the Frechet derivative
    with respect to $\phi$ is

    $$
    D_\phi j[\nu] = 2\operatorname{Re}\bigl(
        \overline{u(x_0)}\,(\alpha\mathcal D - i\eta\mathcal S)\nu(x_0)
    \bigr).
    $$

    Under the $L^2$ sesquilinear form $\langle f,g\rangle = \int f\,\overline g$,
    the Riesz representation is

    $$
    \operatorname{grad}_\phi j(\tau)
    = 2\,u(x_0)\,
      \overline{\bigl(\alpha\widetilde{\mathcal D}(x_0,\tau)
                    - i\eta\,\widetilde{\mathcal S}(x_0,\tau)\bigr)},
    $$

    where

    $$
    \widetilde{\mathcal S}(x_0,\tau) := G(x_0, x(\tau))\,|x'(\tau)|,
    \qquad
    \widetilde{\mathcal D}(x_0,\tau) := n(\tau)\cdot\nabla_y G(x_0, x(\tau))\,|x'(\tau)|
    $$

    are the kernels of $\mathrm{SL}_x$, $\mathrm{DL}_x$
    with jacobian multiplied, evaluated at $x_0$.
    The conjugate involves Hankel functions of the second kind
    $H_n^{(2)} = \overline{H_n^{(1)}}$.

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

    """
    xp = array_namespace(point, k, alpha, eta)
    u_sq = xp.squeeze(field_value)

    def grad(t_in: Array) -> Array:
        dlp = dlp_kernel(point, shape_x=shape.x, shape_dx=shape.dx, k=k, tau=t_in)
        slp = slp_kernel(point, shape_x=shape.x, shape_dx=shape.dx, k=k, tau=t_in)
        conj_k = alpha * xp.conj(dlp) + 1j * eta * xp.conj(slp)
        return 2.0 * u_sq * conj_k

    return grad
