from __future__ import annotations

from typing import Any

import pytest
from array_api.latest import Array
from ie_circle import CircleShape, SympyShape, trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._adjoint import objective_derivative
from biem_helmholtz_2d._objective import grad_phi_scattered_field


def test_adjoint_central_derivative(
    xp: Any,
    device: Any,
    dtype: Any,
) -> None:
    r"""
    Compare adjoint-based shape derivative with central finite differences.

    Objective $J(r) = |u_{\mathrm{scat}}(x_0, r)|^2$ (scattered field at
    $x_0 = (3,3)$).  Perturbation is $\cos(2t)\,\mathbf e_r$.
    """
    import sympy

    from biem_helmholtz_2d.optimization._shape import ParameterShape

    n = 8
    rho = 1.0
    shape = CircleShape(rho)
    k_arr = xp.asarray(1.0, device=device, dtype=dtype)
    a = xp.asarray(1.0, device=device, dtype=dtype)
    e = xp.asarray(0.0, device=device, dtype=dtype)
    x0 = xp.asarray([3.0, 3.0], device=device, dtype=dtype)
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    m = 2

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
    u_scat = near_field(phi, x0[None], k=k_arr, shape=shape, n=n, alpha=a, eta=e)
    grad_phi_j = grad_phi_scattered_field(x0[None], u_scat, shape=shape, k=k_arr, alpha=a, eta=e)

    sym_t = sympy.Symbol("t")
    h_shape = SympyShape(
        sympy.cos(m * sym_t) * sympy.cos(sym_t),
        sympy.cos(m * sym_t) * sympy.sin(sym_t),
        sym_t,
    )

    def dr_g_func(t_in: Array) -> Array:
        xt = shape.x(t_in)
        grad = xp.stack(
            [1j * k_arr * xp.exp(1j * k_arr * xt[..., 0]), xp.zeros_like(xt[..., 0])],
            axis=-1,
        )
        return -xp.sum(grad * h_shape.x(t_in), axis=-1)

    dr_g_vals = dr_g_func(t)

    eps_fd_r = 1e-5
    dr_expr = sympy.cos(m * sym_t)

    def make_shape(eps_pert: float) -> SympyShape:
        r_expr = rho + eps_pert * dr_expr
        return SympyShape(
            r_expr * sympy.cos(sym_t),
            r_expr * sympy.sin(sym_t),
            sym_t,
        )

    shape_plus = make_shape(eps_fd_r)
    shape_minus = make_shape(-eps_fd_r)

    u_plus = near_field(phi, x0[None], k=k_arr, shape=shape_plus, n=n, alpha=a, eta=e)
    u_minus = near_field(phi, x0[None], k=k_arr, shape=shape_minus, n=n, alpha=a, eta=e)
    dr_A_phi = (u_plus - u_minus) / (2 * eps_fd_r)
    dr_j_val = 2.0 * xp.real(xp.conj(u_scat) * dr_A_phi).squeeze()

    dr_adj = objective_derivative(
        k=k_arr,
        shape=shape,
        alpha=a,
        eta=e,
        n=n,
        phi=phi,
        grad_phi_j=grad_phi_j,
        dr_j=dr_j_val,
        dr_g=dr_g_vals,
        h_shape=h_shape,
    )

    def J(params: Array) -> Array:
        ps = ParameterShape(params)
        pp = scattering_dirichlet(
            k=k_arr,
            shape=ps,
            incident_field=incident_field,
            alpha=a,
            eta=e,
            n=n,
        )
        up = near_field(pp, x0[None], k=k_arr, shape=ps, n=n, alpha=a, eta=e)
        return xp.sum(xp.abs(up) ** 2)

    p0 = xp.zeros(max(m + 1, 3) * 2 - 1, device=device, dtype=dtype)
    p0[0] = rho

    eps_fd = 1e-4
    p_plus = p0.copy()
    p_plus[m] += eps_fd
    p_minus = p0.copy()
    p_minus[m] -= eps_fd
    dr_num = (J(p_plus) - J(p_minus)) / (2 * eps_fd)

    assert float(dr_adj) == pytest.approx(float(dr_num), rel=1e-5, abs=1e-8)
