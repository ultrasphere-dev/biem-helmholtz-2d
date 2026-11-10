from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from array_api.latest import Array
from ie_circle import CircleShape, Shape, trapezoidal_quadrature

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._adjoint import objective_derivative, solve_adjoint
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import grad_phi_abs2_scattered_field
from biem_helmholtz_2d._potential_inner_derivative import (
    dlp_shape_derivative,
    slp_shape_derivative,
)


@dataclass(frozen=True)
class ShiftedShape:
    shape: Shape
    center: Array

    def x(self, t: Array, /) -> Array:
        return self.shape.x(t) + self.center

    def dx(self, t: Array, /) -> Array:
        return self.shape.dx(t)

    def ddx(self, t: Array, /) -> Array:
        return self.shape.ddx(t)


def remove_trailing_exponent_zeros(s: str, /) -> str:
    return s.replace("E+0", "E+").replace("E-0", "E-").replace("e+0", "e+").replace("e-0", "e-")


@pytest.mark.parametrize("n", [8, 16, 32, 64])
@pytest.mark.parametrize(("n_scatterers", "moving"), [(1, 0), (2, 0), (2, 1)])
def test_adjoint_central_derivative(
    xp: Any,
    shape: Shape,
    shape_h: Shape,
    shape_central_difference: Callable[[float], tuple[Shape, Shape]],
    device: Any,
    dtype: Any,
    n: int,
    n_scatterers: int,
    moving: int,
) -> None:
    r"""
    Compare adjoint-based shape derivative with central finite differences.

    Objective $J(x) = |u_{\mathrm{scat}}(x_0, x)|^2$ (scattered field at
    $x_0 = (3,3)$). Perturb one boundary by $h = \mathtt{shape\_h}$,
    with either one or two disjoint scatterers.
    """
    k_arr = xp.asarray(1, device=device, dtype=dtype)
    alpha = xp.asarray(1, device=device, dtype=dtype)
    eta = xp.asarray(1, device=device, dtype=dtype)
    x0 = xp.asarray([3, 3], device=device, dtype=dtype)
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident_field = plane_wave(k_arr, direction)
    centers = xp.asarray([[0, 0], [6, 0]], device=device, dtype=dtype)
    shapes = [ShiftedShape(shape, centers[i]) for i in range(n_scatterers)]
    perturbations = [shape_h if i == moving else CircleShape(0) for i in range(n_scatterers)]

    phi = scattering_dirichlet(
        k=k_arr, shape=shapes, incident_field=incident_field, alpha=alpha, eta=eta, n=n
    )
    u_scat = near_field(phi, x0, k=k_arr, shape=shapes, n=n, alpha=alpha, eta=eta)
    zero = xp.asarray(0, dtype=dtype, device=device)
    gradients = [
        grad_phi_abs2_scattered_field(
            x0, u_scat, shape=boundary, k=k_arr, alpha=alpha, eta=eta, target=zero
        )
        for boundary in shapes
    ]

    def grad_phi_j(parameter: Array, /) -> Array:
        return xp.stack([gradient(parameter) for gradient in gradients], axis=-1)

    incident_field_grad = plane_wave_grad(k_arr, direction)
    dr_g_vals = xp.stack(
        [
            -xp.sum(incident_field_grad(boundary.x(t)) * perturbation.x(t), axis=-1)
            for boundary, perturbation in zip(shapes, perturbations, strict=True)
        ],
        axis=-1,
    )

    # Compute dr_j analytically using shape derivatives of near-field operators
    ds = slp_shape_derivative(
        x0,
        lambda t: phi(t)[..., moving],
        shape_x=shapes[moving].x,
        shape_dx=shapes[moving].dx,
        h=shape_h.x,
        dh=shape_h.dx,
        k=k_arr,
        n=n,
    )
    dd = dlp_shape_derivative(
        x0,
        lambda t: phi(t)[..., moving],
        shape_x=shapes[moving].x,
        shape_dx=shapes[moving].dx,
        h=shape_h.x,
        dh=shape_h.dx,
        k=k_arr,
        n=n,
    )
    dr_A_phi = alpha * dd - 1j * eta * ds
    dr_j_val = 2 * xp.real(xp.conj(u_scat) * dr_A_phi)

    dr_adj = objective_derivative(
        k=k_arr,
        shape=shapes,
        alpha=alpha,
        eta=eta,
        n=n,
        phi=phi,
        adjoint=solve_adjoint(
            k=k_arr, shape=shapes, alpha=alpha, eta=eta, n=n, grad_phi_j=grad_phi_j
        ),
        dr_j=dr_j_val,
        dr_g=dr_g_vals,
        h_shape=perturbations,
    )

    def objective(s: Shape, /) -> Array:
        varied = [
            ShiftedShape(s if i == moving else shape, centers[i]) for i in range(n_scatterers)
        ]
        pp = scattering_dirichlet(
            k=k_arr, shape=varied, incident_field=incident_field, alpha=alpha, eta=eta, n=n
        )
        up = near_field(pp, x0, k=k_arr, shape=varied, n=n, alpha=alpha, eta=eta)
        return xp.sum(xp.abs(up) ** 2)

    rows: list[dict[str, object]] = []
    dr_adj_float = float(dr_adj)
    dr_num_ref = None

    exponents = range(-1, -11, -1) if n_scatterers == 1 else (-5,)
    for exponent in exponents:
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

    if n_scatterers == 1:
        df = pd.DataFrame(rows, columns=["kind", "val", "diff"])
        csv_name = f"{shape.__class__.__name__}_{shape_h.__class__.__name__}_n{n}.csv"
        path = Path(__file__).parent / "test_adjoint"
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(path / csv_name, index=False)

    assert dr_num_ref is not None
    assert dr_adj_float == pytest.approx(dr_num_ref, rel=1e-5, abs=1e-8)
