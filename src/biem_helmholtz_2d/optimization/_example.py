from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from array_api.latest import Array, ArrayNamespace
from ie_circle import NystromInterpolant, trapezoidal_quadrature
from matplotlib import pyplot as plt
from scipy.optimize import NonlinearConstraint, minimize

from biem_helmholtz_2d._acoustic import (
    near_field,
    plot_near_field,
    plot_near_field_prepare,
    scattering_dirichlet,
)
from biem_helmholtz_2d._adjoint import objective_derivative
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import grad_phi_abs2_scattered_field
from biem_helmholtz_2d._potential_inner_derivative import (
    dlp_shape_derivative,
    slp_shape_derivative,
)
from biem_helmholtz_2d.optimization._shape import ParameterShape


def example_optimization(
    *,
    path: Path,
    xp: ArrayNamespace,
    dtype: Any,
    device: Any,
    k: complex,
    alpha: complex,
    eta: complex,
    n: int,
    n_steps: int = 10,
    n_modes: int = 20,
    alpha_reg: float = 0.1,
    k_reg: int = 3,
    desired_total_field: complex = 0j,
    target_point: tuple[float, float] = (-2, 3),
) -> None:
    r"""
    Example optimization using adjoint method with trust-constr.

    Minimizes $|u_{\mathrm{total}}(x_0) - c|^2$ subject to
    $\sum \sqrt{a_n^2 + b_n^2} \le 1$, where
    $u_{\mathrm{total}} = u_{\mathrm{scat}} + u_{\mathrm{inc}}$.
    The constant Fourier coefficient is fixed to $1$.

    Saves plot-ready data as JSON files in ``path``. Use
    :func:`example_optimization_plot` to generate figures.

    Parameters
    ----------
    path : Path
        Directory to save JSON files.
    n : int
        Maximum order minus $1$.
    xp : ArrayNamespace
        Array API namespace.
    dtype : Any
        Array dtype.
    device : Any
        Array device.
    n_modes : int
        Number of Fourier modes divided by 2.
    n_steps : int
        Number of optimization steps.
    alpha_reg : float
        Hilbertian regularization weight $\alpha$ in the inner product

        $$
        \langle \phi, \psi \rangle_{H_{2\pi}^k}
        = \tfrac{1}{2} a_0(\phi) a_0(\psi)
        + \sum_{m=1}^{\infty} (1 + \alpha m^2)^k
            \bigl(a_m(\phi) a_m(\psi) + b_m(\phi) b_m(\psi)\bigr).
        $$
    k_reg : int
        Sobolev exponent $k$. $H_{2\pi}^3(\mathbb{R}) \subset C_{2\pi}^2(\mathbb{R})$.
    desired_total_field : complex
        Desired total field value $c$ at $x_0$.
    k : complex
        The wave number.
    alpha : complex
        The coupling parameter for the double-layer potential.
    eta : complex
        The coupling parameter for the single-layer potential.
    target_point : tuple[float, float]
        The point $x_0$ at which the total field is minimized.

    """
    path.mkdir(parents=True, exist_ok=True)
    k = xp.asarray(k, device=device, dtype=dtype)
    eta = xp.asarray(eta, device=device, dtype=dtype)
    alpha = xp.asarray(alpha, device=device, dtype=dtype)
    point = xp.asarray(target_point, device=device, dtype=dtype)
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident_field = plane_wave(k, direction)
    incident_field_grad = plane_wave_grad(k, direction)
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)

    # Basis shape perturbations, batched over all 2 * n_modes design directions
    eye = xp.eye(n_modes, dtype=dtype, device=device)
    zeros_n = xp.zeros((n_modes, n_modes), dtype=dtype, device=device)
    zeros_n1 = xp.zeros((n_modes, n_modes + 1), dtype=dtype, device=device)
    eye_cos = xp.eye(n_modes, n_modes + 1, k=1, dtype=dtype, device=device)
    h_all = ParameterShape(
        cos_coefs=xp.concat([eye_cos, zeros_n1], axis=0),
        sin_coefs=xp.concat([zeros_n, eye], axis=0),
    )

    def unpack(x: np.ndarray, /) -> tuple[Array, Array]:
        cos_coefs = xp.concat([
            xp.ones(1, dtype=dtype, device=device),
            xp.asarray(x[:n_modes], dtype=dtype, device=device),
        ])
        sin_coefs = xp.asarray(x[n_modes:], dtype=dtype, device=device)
        return cos_coefs, sin_coefs

    def solve(
        cos_coefs: Array, sin_coefs: Array, /
    ) -> tuple[ParameterShape, NystromInterpolant, Array, Array]:
        shape = ParameterShape(cos_coefs=cos_coefs, sin_coefs=sin_coefs)
        phi = scattering_dirichlet(
            k=k, shape=shape, incident_field=incident_field, alpha=alpha, eta=eta, n=n
        )
        u_scat = near_field(phi, point[None], k=k, shape=shape, n=n, alpha=alpha, eta=eta)
        u_inc = incident_field(point)
        return shape, phi, u_scat, u_inc

    def fun(x: np.ndarray) -> float:
        cos_coefs, sin_coefs = unpack(x)
        _, _, u_scat, u_inc = solve(cos_coefs, sin_coefs)
        u_total = u_scat + u_inc
        return float(xp.sum(xp.abs(u_total - desired_total_field) ** 2))

    def jac(x: np.ndarray) -> np.ndarray:
        cos_coefs, sin_coefs = unpack(x)
        shape, phi, u_scat, u_inc = solve(cos_coefs, sin_coefs)
        target = desired_total_field - u_inc
        grad_phi_j = grad_phi_abs2_scattered_field(
            point[None], u_scat, shape=shape, k=k, alpha=alpha, eta=eta, target=target
        )

        incident_grad_at_shape = incident_field_grad(shape.x(t))
        dr_g = -xp.sum(incident_grad_at_shape * h_all.x(t), axis=-1)
        slp_deriv = slp_shape_derivative(
            point[None],
            phi,
            shape_x=shape.x,
            shape_dx=shape.dx,
            h=h_all.x,
            dh=h_all.dx,
            k=k,
            n=n,
        )
        dlp_deriv = dlp_shape_derivative(
            point[None],
            phi,
            shape_x=shape.x,
            shape_dx=shape.dx,
            h=h_all.x,
            dh=h_all.dx,
            k=k,
            n=n,
        )
        dr_j = 2 * xp.real(
            (xp.conj(u_scat) - xp.conj(target)) * (alpha * dlp_deriv - 1j * eta * slp_deriv)
        )
        gradient = objective_derivative(
            k=k,
            shape=shape,
            alpha=alpha,
            eta=eta,
            n=n,
            phi=phi,
            grad_phi_j=grad_phi_j,
            dr_j=dr_j,
            dr_g=dr_g,
            h_shape=h_all,
        )
        m = xp.arange(1, n_modes + 1, dtype=dtype, device=device)
        weights = (1 + alpha_reg * m**2) ** k_reg
        gradient = gradient / xp.concat([weights, weights])
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

    constraint = NonlinearConstraint(constraint_fun, -np.inf, 1, jac=constraint_jac)

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
        options={"verbose": 3, "maxiter": n_steps},
    )

    cos_coefs_opt, sin_coefs_opt = unpack(result.x)
    shape_opt = ParameterShape(cos_coefs=cos_coefs_opt, sin_coefs=sin_coefs_opt)

    t_plot = np.linspace(0, 2 * np.pi, 10000)
    t_arr = xp.asarray(t_plot, dtype=dtype, device=device)
    x_plot = np.asarray(shape_opt.x(t_arr), device="cpu")

    # Optimization history
    (path / "optimization_history.json").write_bytes(
        orjson.dumps({
            "val_hist": val_hist,
            "k": {"real": float(k.real), "imag": float(k.imag)},
            "n": n,
            "alpha_reg": alpha_reg,
            "k_reg": k_reg,
            "n_modes": n_modes,
            "n_steps": n_steps,
            "target_point": target_point,
            "desired_total_field": {
                "real": float(desired_total_field.real),
                "imag": float(desired_total_field.imag),
            },
            "alpha": {"real": float(alpha.real), "imag": float(alpha.imag)},
            "eta": {"real": float(eta.real), "imag": float(eta.imag)},
            "final_parameters": {
                "cos_coefs": cos_coefs_opt.tolist(),
                "sin_coefs": sin_coefs_opt.tolist(),
            },
        })
    )

    # Optimized shape
    (path / "optimized_shape.json").write_bytes(
        orjson.dumps(
            {
                "x": xp.ascontiguousarray(x_plot[:, 0]),
                "y": xp.ascontiguousarray(x_plot[:, 1]),
            },
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
    )

    # Near-field data
    density_opt = scattering_dirichlet(
        k=k, shape=shape_opt, incident_field=incident_field, alpha=alpha, eta=eta, n=n
    )
    field_data = plot_near_field_prepare(
        density_opt,
        incident_field,
        xlim=(-4, 4),
        ylim=(-4, 4),
        k=k,
        shape=shape_opt,
        n=n,
        alpha=alpha,
        eta=eta,
        n_plot=200,
        isin_shape_n_quad=500,
        isin_shape_tol=1e-5,
    )
    (path / "optimized_near_field.json").write_bytes(
        orjson.dumps(field_data, option=orjson.OPT_SERIALIZE_NUMPY)
    )


def example_optimization_plot(path: pathlib.Path) -> None:
    """
    Generate plots from JSON data saved by :func:`example_optimization`.

    Parameters
    ----------
    path : pathlib.Path
        Directory containing ``optimization_history.json``,
        ``optimized_shape.json``, and ``optimized_near_field.json``.

    """
    # Load data
    history = json.loads((path / "optimization_history.json").read_text())
    shape_data = json.loads((path / "optimized_shape.json").read_text())
    field_data = json.loads((path / "optimized_near_field.json").read_text())
    alpha = history["alpha_reg"]

    # Optimization history plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(history["val_hist"])
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Optimization history (alpha={history['alpha_reg']})")
    fig.tight_layout()
    fig.savefig(path / "optimization_history.svg")
    plt.close(fig)

    # Magnitude of coefficients plot
    fig, ax = plt.subplots(figsize=(4, 3))
    sin_coefs = np.asarray(history["final_parameters"]["sin_coefs"])
    cos_coefs = np.asarray(history["final_parameters"]["cos_coefs"])[1:]
    n_modes = len(sin_coefs)
    m = np.arange(1, n_modes + 1)
    ax.plot(m, np.abs(cos_coefs), "o-", label="cosine coefficients")
    ax.plot(m, np.abs(sin_coefs), "o-", label="sine coefficients")
    ax.set_yscale("log")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Magnitude")
    ax.set_title("Optimized Fourier coefficients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / f"optimized_coefficients_{alpha}.svg")
    plt.close(fig)

    # Optimized shape plot
    fig, ax = plt.subplots()
    ax.plot(shape_data["x"], shape_data["y"])
    ax.set_aspect("equal")
    ax.set_title("Optimized shape")
    fig.tight_layout()
    fig.savefig(path / f"optimized_shape_{alpha}.svg")
    plt.close(fig)

    # Near-field plot
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
    plot_near_field(
        field_data,
        ax_utot_re=ax[0],
        ax_utot_im=None,
        ax_utot_abs=ax[1],
    )
    for a in ax:
        a.plot(
            history["target_point"][0],
            history["target_point"][1],
            "X",
            markersize=15,
            markerfacecolor="black",
            markeredgewidth=2,
            markeredgecolor="white",
            label="Point to minimize",
        )
        a.legend()
    fig.tight_layout()
    fig.savefig(path / f"optimized_near_field_{alpha}.svg")
    plt.close(fig)
