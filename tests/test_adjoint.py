from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from array_api.latest import Array
from ie_circle import Shape, trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._adjoint import objective_derivative
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import grad_phi_scattered_field


def remove_trailing_exponent_zeros(s: str, /) -> str:
    return s.replace("E+0", "E+").replace("E-0", "E-").replace("e+0", "e+").replace("e-0", "e-")


def test_adjoint_central_derivative(
    xp: Any,
    shape: Shape,
    shape_h: Shape,
    shape_central_difference: Callable[[float], tuple[Shape, Shape]],
    device: Any,
    dtype: Any,
) -> None:
    r"""
    Compare adjoint-based shape derivative with central finite differences.

    Objective $J(x) = |u_{\mathrm{scat}}(x_0, x)|^2$ (scattered field at
    $x_0 = (3,3)$).  Perturbation is $h = \mathtt{shape\_h}$.
    """
    n = 8
    k_arr = xp.asarray(1.0, device=device, dtype=dtype)
    a = xp.asarray(1.0, device=device, dtype=dtype)
    e = xp.asarray(0.0, device=device, dtype=dtype)
    x0 = xp.asarray([3.0, 3.0], device=device, dtype=dtype)
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    direction = xp.asarray([1.0, 0.0], device=device, dtype=dtype)
    incident_field = plane_wave(k_arr, direction)

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

    incident_field_grad = plane_wave_grad(k_arr, direction)
    dr_g_vals = -xp.sum(incident_field_grad(shape.x(t)) * shape_h.x(t), axis=-1)

    eps = 1e-5

    shape_p, shape_m = shape_central_difference(eps)
    u_plus = near_field(phi, x0[None], k=k_arr, shape=shape_p, n=n, alpha=a, eta=e)
    u_minus = near_field(phi, x0[None], k=k_arr, shape=shape_m, n=n, alpha=a, eta=e)
    dr_A_phi = (u_plus - u_minus) / (2 * eps)
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
        h_shape=shape_h,
    )

    def objective(s: Shape) -> Array:
        pp = scattering_dirichlet(
            k=k_arr,
            shape=s,
            incident_field=incident_field,
            alpha=a,
            eta=e,
            n=n,
        )
        up = near_field(pp, x0[None], k=k_arr, shape=s, n=n, alpha=a, eta=e)
        return xp.sum(xp.abs(up) ** 2)

    rows: list[dict[str, object]] = []
    dr_adj_float = float(dr_adj)
    dr_num_ref = None

    for exponent in range(-1, -11, -1):
        eps_val = 10**exponent
        shape_p, shape_m = shape_central_difference(eps_val)
        j_plus = objective(shape_p)
        j_minus = objective(shape_m)
        dr_num_val = float((j_plus - j_minus) / (2 * eps_val))
        diff = abs(dr_adj_float - dr_num_val)
        rows.append({
            "kind": f"1e{exponent}",
            "val": remove_trailing_exponent_zeros(f"{dr_num_val:.12e}"),
            "diff": remove_trailing_exponent_zeros(f"{diff:.1e}"),
        })
        if exponent == -5:
            dr_num_ref = dr_num_val

    rows.append({
        "kind": "None",
        "val": remove_trailing_exponent_zeros(f"{dr_adj_float:.12e}"),
        "diff": None,
    })

    df = pd.DataFrame(rows, columns=["kind", "val", "diff"])
    csv_name = f"test_adjoint_{shape.__class__.__name__}_{shape_h.__class__.__name__}.csv"
    df.to_csv(Path(__file__).parent / csv_name, index=False)

    assert dr_num_ref is not None
    assert dr_adj_float == pytest.approx(dr_num_ref, rel=1e-5, abs=1e-8)
