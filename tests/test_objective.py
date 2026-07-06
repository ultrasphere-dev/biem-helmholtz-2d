from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from array_api.latest import Array
from ie_circle import CircleShape, trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._objective import grad_phi_scattered_field


@pytest.mark.parametrize("m", [1, 3])
def test_grad_phi_central_derivative(
    xp: Any,
    device: Any,
    dtype: Any,
    m: int,
) -> None:
    r"""
    Verify $\operatorname{grad}_\phi j$ by central FD on $j(\phi)=|u(x_0)|^2$.

    Perturbation $v(t) = \cos(m t)$.
    """
    n = 8
    shape = CircleShape(1.0)
    k_arr = xp.asarray(1.0, device=device, dtype=dtype)
    a = xp.asarray(1.0, device=device, dtype=dtype)
    e = xp.asarray(0.0, device=device, dtype=dtype)
    x0 = xp.asarray([3.0, 3.0], device=device, dtype=dtype)
    t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    wt = w[0]

    def incident_field(x: Array) -> Array:
        return xp.exp(1j * k_arr * x[..., 0])

    phi = scattering_dirichlet(
        k=k_arr,
        shape=shape,
        incident_field=incident_field,
        alpha=a,
        eta=e,
        n=n,
    )
    u = near_field(phi, x0[None], k=k_arr, shape=shape, n=n, alpha=a, eta=e)
    grad_phi_j = grad_phi_scattered_field(x0[None], u, shape=shape, k=k_arr, alpha=a, eta=e)

    def v_func(t_in: Array) -> Array:
        return xp.cos(m * t_in)

    v_t = v_func(t)
    gj_t = grad_phi_j(t)
    inner = xp.sum(gj_t * v_t * wt)

    eps_fd = 1e-5

    def j_of_phi(phi_pert: Callable[[Array], Array]) -> Array:
        up = near_field(phi_pert, x0[None], k=k_arr, shape=shape, n=n, alpha=a, eta=e)
        return xp.sum(xp.abs(up) ** 2)

    class _PerturbedDensity:
        def __init__(self, eps_pert: float) -> None:
            self._eps = eps_pert

        def __call__(self, t_in: Array) -> Array:
            return phi(t_in) + self._eps * v_func(t_in)

    j_plus = j_of_phi(_PerturbedDensity(eps_fd))
    j_minus = j_of_phi(_PerturbedDensity(-eps_fd))
    fd_deriv = (j_plus - j_minus) / (2 * eps_fd)

    tol = 5e-2 * abs(fd_deriv) + 5e-2 * abs(inner) + 1e-4
    assert xp.abs(inner - fd_deriv) < tol, (
        f"Riesz inner {inner:.6f}  FD {fd_deriv:.6f}  diff {abs(inner - fd_deriv):.6f}"
    )
