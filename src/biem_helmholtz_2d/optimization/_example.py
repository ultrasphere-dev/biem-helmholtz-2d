from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Any

import numpy as np
from array_api.latest import Array, ArrayNamespace
from ie_circle import NystromInterpolant, trapezoidal_quadrature
from matplotlib import pyplot as plt
from scipy.optimize import NonlinearConstraint, minimize

from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._adjoint import objective_derivative
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import grad_phi_abs2_scattered_field
from biem_helmholtz_2d._potential_inner_derivative import (
    dlp_shape_derivative,
    slp_shape_derivative,
)
from biem_helmholtz_2d.optimization._shape import ParameterShape


def example_optimization(*, xp: ArrayNamespace, dtype: Any, device: Any) -> None:
    """
    Example optimization using adjoint method with trust-constr.

    Minimizes |u_scat(x_0)|^2 subject to sum sqrt(a_n^2 + b_n^2) <= 1.
    The constant Fourier coefficient is fixed to 1.

    """
    n = 8
    n_modes = 4
    k = xp.asarray(1.0, device=device, dtype=dtype)
    eta = xp.asarray(0.0, device=device, dtype=dtype)
    alpha = xp.asarray(1.0, device=device, dtype=dtype)
    point = xp.asarray([3.0, 3.0], device=device, dtype=dtype)
    direction = xp.asarray([1.0, 0.0], device=device, dtype=dtype)
    incident_field = plane_wave(k, direction)
    incident_field_grad = plane_wave_grad(k, direction)
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)

    def unpack(x: np.ndarray, /) -> tuple[Array, Array]:
        cos_coefs = xp.concat([
            xp.ones(1, dtype=dtype, device=device),
            xp.asarray(x[:n_modes], dtype=dtype, device=device),
        ])
        sin_coefs = xp.asarray(x[n_modes:], dtype=dtype, device=device)
        return cos_coefs, sin_coefs

    def solve(
        cos_coefs: Array, sin_coefs: Array, /
    ) -> tuple[ParameterShape, NystromInterpolant, Array]:
        shape = ParameterShape(cos_coefs=cos_coefs, sin_coefs=sin_coefs)
        phi = scattering_dirichlet(
            k=k, shape=shape, incident_field=incident_field, alpha=alpha, eta=eta, n=n
        )
        u_scat = near_field(phi, point[None], k=k, shape=shape, n=n, alpha=alpha, eta=eta)
        return shape, phi, u_scat

    def fun(x: np.ndarray) -> float:
        cos_coefs, sin_coefs = unpack(x)
        _, _, u_scat = solve(cos_coefs, sin_coefs)
        return float(xp.sum(xp.abs(u_scat) ** 2))

    def jac(x: np.ndarray) -> np.ndarray:
        cos_coefs, sin_coefs = unpack(x)
        shape, phi, u_scat = solve(cos_coefs, sin_coefs)
        grad_phi_j = grad_phi_abs2_scattered_field(
            point[None], u_scat, shape=shape, k=k, alpha=alpha, eta=eta
        )

        # Basis shape perturbations, batched over all design directions
        h_cos = ParameterShape(
            cos_coefs=xp.eye(n_modes, n_modes + 1, k=1, dtype=dtype, device=device),
            sin_coefs=xp.zeros((n_modes, n_modes), dtype=dtype, device=device),
        )
        h_sin = ParameterShape(
            cos_coefs=xp.zeros((n_modes, n_modes + 1), dtype=dtype, device=device),
            sin_coefs=xp.eye(n_modes, dtype=dtype, device=device),
        )
        incident_grad_at_shape = incident_field_grad(shape.x(t))

        def derivative(h_shape: ParameterShape, /) -> Array:
            dr_g = -xp.sum(incident_grad_at_shape * h_shape.x(t), axis=-1)
            slp_deriv = slp_shape_derivative(
                point[None],
                phi,
                shape_x=shape.x,
                shape_dx=shape.dx,
                h=h_shape.x,
                dh=h_shape.dx,
                k=k,
                n=n,
            )
            dlp_deriv = dlp_shape_derivative(
                point[None],
                phi,
                shape_x=shape.x,
                shape_dx=shape.dx,
                h=h_shape.x,
                dh=h_shape.dx,
                k=k,
                n=n,
            )
            dr_j = 2.0 * xp.real(xp.conj(u_scat) * (alpha * dlp_deriv - 1j * eta * slp_deriv))
            return objective_derivative(
                k=k,
                shape=shape,
                alpha=alpha,
                eta=eta,
                n=n,
                phi=phi,
                grad_phi_j=grad_phi_j,
                dr_j=dr_j,
                dr_g=dr_g,
                h_shape=h_shape,
            )

        gradient = xp.concat([derivative(h_cos), derivative(h_sin)])
        return np.asarray(gradient, device="cpu")

    def constraint_fun(x: np.ndarray) -> float:
        cos_part = x[:n_modes]
        sin_part = x[n_modes:]
        return float(np.sum(np.sqrt(cos_part**2 + sin_part**2)))

    def constraint_jac(x: np.ndarray) -> np.ndarray:
        cos_part = x[:n_modes]
        sin_part = x[n_modes:]
        r = np.sqrt(cos_part**2 + sin_part**2)
        gradient = np.zeros_like(x)
        np.divide(cos_part, r, out=gradient[:n_modes], where=r > 0)
        np.divide(sin_part, r, out=gradient[n_modes:], where=r > 0)
        return gradient

    constraint = NonlinearConstraint(constraint_fun, -np.inf, 1.0, jac=constraint_jac)

    path = pathlib.Path(f"optimization/{datetime.now().strftime('%Y%m%d_%H%M%S')}_k{float(k)}_n{n}")
    path.mkdir(parents=True, exist_ok=True)

    val_hist = []

    def callback(intermediate_result: Any) -> None:
        val_hist.append(intermediate_result.fun)

    result = minimize(
        fun,
        np.zeros(2 * n_modes),
        method="trust-constr",
        jac=jac,
        constraints=constraint,
        callback=callback,
        options={"verbose": 1, "maxiter": 50},
    )

    cos_coefs_opt, sin_coefs_opt = unpack(result.x)
    shape_opt = ParameterShape(cos_coefs=cos_coefs_opt, sin_coefs=sin_coefs_opt)

    fig, ax = plt.subplots()
    ax.plot(val_hist)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Optimization history (k={float(k)}, n={n})")
    fig.savefig(path / "optimization_history.png")

    fig, ax = plt.subplots()
    t_plot = np.linspace(0, 2 * np.pi, 200)
    t_arr = xp.asarray(t_plot, dtype=dtype, device=device)
    x_plot = np.asarray(shape_opt.x(t_arr), device="cpu")
    ax.plot(x_plot[:, 0], x_plot[:, 1])
    ax.set_aspect("equal")
    ax.set_title("Optimized shape")
    fig.savefig(path / "optimized_shape.png")
