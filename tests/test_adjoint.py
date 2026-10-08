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
from biem_helmholtz_2d._objective import grad_phi_abs2_scattered_field
from biem_helmholtz_2d._potential_inner_derivative import (
    dlp_shape_derivative,
    slp_shape_derivative,
)


def remove_trailing_exponent_zeros(s: str, /) -> str:
    return s.replace("E+0", "E+").replace("E-0", "E-").replace("e+0", "e+").replace("e-0", "e-")


@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_adjoint_central_derivative(
    xp: Any,
    shape: Shape,
    shape_h: Shape,
    shape_central_difference: Callable[[float], tuple[Shape, Shape]],
    device: Any,
    dtype: Any,
    n: int,
) -> None:
    r"""
    Compare adjoint-based shape derivative with central finite differences.

    Objective $J(x) = |u_{\mathrm{scat}}(x_0, x)|^2$ (scattered field at
    $x_0 = (3,3)$).  Perturbation is $h = \mathtt{shape\_h}$.
    """
    k_arr = xp.asarray(1, device=device, dtype=dtype)
    alpha = xp.asarray(1, device=device, dtype=dtype)
    eta = xp.asarray(1, device=device, dtype=dtype)
    x0 = xp.asarray([3, 3], device=device, dtype=dtype)
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident_field = plane_wave(k_arr, direction)

    phi = scattering_dirichlet(
        k=k_arr,
        shape=shape,
        incident_field=incident_field,
        alpha=alpha,
        eta=eta,
        n=n,
    )
    u_scat = near_field(phi, x0[None], k=k_arr, shape=shape, n=n, alpha=alpha, eta=eta)
    zero = xp.asarray(0, dtype=dtype, device=device)
    grad_phi_j = grad_phi_abs2_scattered_field(
        x0[None], u_scat, shape=shape, k=k_arr, alpha=alpha, eta=eta, target=zero
    )

    incident_field_grad = plane_wave_grad(k_arr, direction)
    dr_g_vals = -xp.sum(incident_field_grad(shape.x(t)) * shape_h.x(t), axis=-1)

    # Compute dr_j analytically using shape derivatives of near-field operators
    ds = slp_shape_derivative(
        x0[None],
        phi,
        shape_x=shape.x,
        shape_dx=shape.dx,
        h=shape_h.x,
        dh=shape_h.dx,
        k=k_arr,
        n=n,
    )
    dd = dlp_shape_derivative(
        x0[None],
        phi,
        shape_x=shape.x,
        shape_dx=shape.dx,
        h=shape_h.x,
        dh=shape_h.dx,
        k=k_arr,
        n=n,
    )
    dr_A_phi = xp.squeeze(alpha * dd - 1j * eta * ds)
    dr_j_val = 2 * xp.real(xp.conj(u_scat) * dr_A_phi).squeeze()

    dr_adj = objective_derivative(
        k=k_arr,
        shape=shape,
        alpha=alpha,
        eta=eta,
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
            alpha=alpha,
            eta=eta,
            n=n,
        )
        up = near_field(pp, x0[None], k=k_arr, shape=s, n=n, alpha=alpha, eta=eta)
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
    csv_name = f"{shape.__class__.__name__}_{shape_h.__class__.__name__}_n{n}.csv"
    path = Path(__file__).parent / "test_adjoint"
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / csv_name, index=False)

    assert dr_num_ref is not None
    assert dr_adj_float == pytest.approx(dr_num_ref, rel=1e-5, abs=1e-8)
