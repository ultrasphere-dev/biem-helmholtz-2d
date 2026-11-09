from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from array_api.latest import Array
from ie_circle import Shape, trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._incident import plane_wave
from biem_helmholtz_2d._objective import grad_phi_abs2_scattered_field


@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize("target_val", [0, 1 + 0.5j])
def test_grad_phi_central_derivative(
    xp: Any,
    shape: Shape,
    device: Any,
    dtype: Any,
    m: int,
    target_val: complex,
) -> None:
    r"""
    Verify $\operatorname{grad}_\phi j$ by central FD on $j(\phi)=|u(x_0) - c|^2$.

    Perturbation $v(t) = \cos(m t)$.
    """
    n = 8
    k_arr = xp.asarray(1, device=device, dtype=dtype)
    alpha = xp.asarray(1, device=device, dtype=dtype)
    eta = xp.asarray(1, device=device, dtype=dtype)
    x0 = xp.asarray([3, 3], device=device, dtype=dtype)
    t, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    wt = w[0]
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident_field = plane_wave(k_arr, direction)

    density = scattering_dirichlet(
        k=k_arr,
        shape=shape,
        incident_field=incident_field,
        alpha=alpha,
        eta=eta,
        n=n,
    )
    u = near_field(density, x0[None], k=k_arr, shape=shape, n=n, alpha=alpha, eta=eta)

    target_arr = xp.asarray(target_val, dtype=xp.result_type(dtype, 1j), device=device)

    grad_phi_j = grad_phi_abs2_scattered_field(
        x0[None], u, shape=shape, k=k_arr, alpha=alpha, eta=eta, target=target_arr
    )

    def density_perb(t_in: Array) -> Array:
        return xp.cos(m * t_in)

    density_perb_t = density_perb(t)
    gj_t = grad_phi_j(t)
    actual = xp.real(xp.sum(gj_t * xp.conj(density_perb_t) * wt))

    def j_of_phi(density_: Callable[[Array], Array], /) -> Array:
        u = near_field(density_, x0[None], k=k_arr, shape=shape, n=n, alpha=alpha, eta=eta)
        return xp.sum(xp.abs(u - target_arr) ** 2)

    eps = 1e-5

    def density_plus(t_in: Array) -> Array:
        return density(t_in) + eps * density_perb(t_in)[..., None]

    def density_minus(t_in: Array) -> Array:
        return density(t_in) - eps * density_perb(t_in)[..., None]

    j_plus = j_of_phi(density_plus)
    j_minus = j_of_phi(density_minus)
    expected = (j_plus - j_minus) / (2 * eps)

    assert float(actual) == pytest.approx(float(expected), rel=1e-6, abs=1e-8)
