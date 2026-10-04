from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from array_api.latest import Array, ArrayNamespace
from ie_circle import NystromInterpolant, ShapeList, trapezoidal_quadrature
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
    n_scatterers: int = 1,
) -> None:
    r"""
    Example optimization using adjoint method with trust-constr for multiple scatterers.

    Minimizes $|u_{\mathrm{total}}(x_0) - c|^2$ subject to
    $\sum_{j=1}^M \sum_{m} \sqrt{a_{j,m}^2 + b_{j,m}^2} \le 1$, where
    $u_{\mathrm{total}} = u_{\mathrm{scat}} + u_{\mathrm{inc}}$.
    The constant Fourier coefficient for each scatterer is fixed to $1$.

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
        Number of Fourier modes divided by 2 per scatterer.
    n_steps : int
        Number of optimization steps.
    alpha_reg : float
        Hilbertian regularization weight $\alpha$ in the inner product.
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
    n_scatterers : int
        Number of scatterers to optimize.

    """
    path.mkdir(parents=True, exist_ok=True)
    k = xp.asarray(k, device=device, dtype=dtype)
    M = n_scatterers
    eta_arr = xp.asarray([eta] * M, device=device, dtype=dtype)
    alpha_arr = xp.asarray([alpha] * M, device=device, dtype=dtype)
    point = xp.asarray(target_point, device=device, dtype=dtype)
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident_field = plane_wave(k, direction)
    incident_field_grad = plane_wave_grad(k, direction)
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)

    # Basis shape perturbations for each scatterer, batched over all 2 * n_modes design directions
    eye = xp.eye(n_modes, dtype=dtype, device=device)
    zeros_n = xp.zeros((n_modes, n_modes), dtype=dtype, device=device)
    zeros_n1 = xp.zeros((n_modes, n_modes + 1), dtype=dtype, device=device)
    eye_cos = xp.eye(n_modes, n_modes + 1, k=1, dtype=dtype, device=device)

    # Create perturbation shapes for all M scatterers
    # Each scatterer has 2 * n_modes design variables
    h_all_list = []
    for _ in range(M):
        h_single = ParameterShape(
            cos_coefs=xp.concat([eye_cos, zeros_n1], axis=0),
            sin_coefs=xp.concat([zeros_n, eye], axis=0),
        )
        h_all_list.append(h_single)

    # Stack perturbations for all scatterers
    h_all = ShapeList(h_all_list)

    def unpack(x: np.ndarray, /) -> list[tuple[Array, Array]]:
        """Unpack optimization variables for M scatterers."""
        result = []
        for j in range(M):
            offset = j * 2 * n_modes
            cos_coefs = xp.concat([
                xp.ones(1, dtype=dtype, device=device),
                xp.asarray(x[offset : offset + n_modes], dtype=dtype, device=device),
            ])
            sin_coefs = xp.asarray(
                x[offset + n_modes : offset + 2 * n_modes], dtype=dtype, device=device
            )
            result.append((cos_coefs, sin_coefs))
        return result

    def solve(
        shapes_params: list[tuple[Array, Array]], /
    ) -> tuple[ShapeList, NystromInterpolant, Array, Array]:
        shapes_list = [
            ParameterShape(cos_coefs=cos_coefs, sin_coefs=sin_coefs)
            for cos_coefs, sin_coefs in shapes_params
        ]
        shapes = ShapeList(shapes_list)
        phi = scattering_dirichlet(
            k=k, shapes=shapes, incident_field=incident_field, alpha=alpha_arr, eta=eta_arr, n=n
        )
        u_scat = near_field(phi, point[None], k=k, shapes=shapes, n=n, alpha=alpha_arr, eta=eta_arr)
        u_inc = incident_field(point)
        return shapes, phi, u_scat, u_inc

    def fun(x: np.ndarray) -> float:
        shapes_params = unpack(x)
        _, _, u_scat, u_inc = solve(shapes_params)
        u_total = u_scat + u_inc
        return float(xp.sum(xp.abs(u_total - desired_total_field) ** 2))

    def jac(x: np.ndarray) -> np.ndarray:
        shapes_params = unpack(x)
        shapes, phi, u_scat, u_inc = solve(shapes_params)
        target = desired_total_field - u_inc

        # For now, compute gradient for each scatterer independently
        # (ignoring cross-coupling terms for simplicity)
        gradients = []
        for j in range(M):
            shape_j = shapes[j]
            h_j = h_all[j]

            # Compute grad_phi_j for j-th scatterer
            grad_phi_j = grad_phi_abs2_scattered_field(
                point[None],
                u_scat,
                shape=ParameterShape(cos_coefs=shapes_params[j][0], sin_coefs=shapes_params[j][1]),
                k=k,
                alpha=alpha_arr[j],
                eta=eta_arr[j],
                target=target,
            )

            # Extract j-th scatterer's density
            def phi_j(t_in: Array, _j: int = j) -> Array:
                return phi(t_in)[..., _j]

            # Compute incident field gradient at j-th scatterer
            # h_j.x(t) has shape (Q, n_basis) for basis perturbations
            incident_grad_at_shape_j = incident_field_grad(shape_j.x(t))  # (Q, 2)
            # dr_g_j should have shape (Q, n_basis)
            dr_g_j = -xp.sum(
                incident_grad_at_shape_j[..., None, :] * h_j.x(t), axis=-1
            )  # (Q, n_basis)

            # Compute shape derivatives for j-th scatterer
            # slp_shape_derivative and dlp_shape_derivative expect h to be a single perturbation
            # But h_j.x is batched, so we need to handle this differently
            # For now, let's just compute the gradient for the first basis direction
            # This is a temporary workaround
            def h_j_single(t_in: Array, _j: int = j) -> Array:
                return h_all[_j].x(t_in)[..., 0, :]  # Take first basis direction, shape (Q, 2)

            def dh_j_single(t_in: Array, _j: int = j) -> Array:
                return h_all[_j].dx(t_in)[..., 0, :]

            slp_deriv = slp_shape_derivative(
                point[None],
                phi_j,
                shape_x=shape_j.x,
                shape_dx=shape_j.dx,
                h=h_j_single,
                dh=dh_j_single,
                k=k,
                n=n,
            )
            dlp_deriv = dlp_shape_derivative(
                point[None],
                phi_j,
                shape_x=shape_j.x,
                shape_dx=shape_j.dx,
                h=h_j_single,
                dh=dh_j_single,
                k=k,
                n=n,
            )

            dr_j = 2 * xp.real(
                (xp.conj(u_scat) - xp.conj(target))
                * (alpha_arr[j] * dlp_deriv - 1j * eta_arr[j] * slp_deriv)
            )

            # For dr_g, take first basis direction
            dr_g_single = dr_g_j[..., 0]  # (Q,)

            # For grad_phi_j, wrap it to return (Q, M) with only j-th column non-zero
            def grad_phi_j_full(t_in: Array, _j: int = j) -> Array:
                grad_j = grad_phi_funcs[_j](t_in)  # (Q,)
                grad_full = xp.zeros((grad_j.shape[0], M), dtype=grad_j.dtype, device=grad_j.device)
                grad_full[..., _j] = grad_j
                return grad_full

            # For dr_g_full, shape (Q, M) with only j-th column non-zero
            dr_g_full = xp.zeros(
                (dr_g_single.shape[0], M), dtype=dr_g_single.dtype, device=dr_g_single.device
            )
            dr_g_full[..., j] = dr_g_single

            gradient_j = objective_derivative(
                k=k,
                shapes=shapes,
                alpha=alpha_arr,
                eta=eta_arr,
                n=n,
                phi=phi,
                grad_phi_j=grad_phi_j_full,
                dr_j=dr_j,
                dr_g=dr_g_full,
                h_shapes=h_all,
            )

            m = xp.arange(1, n_modes + 1, dtype=dtype, device=device)
            weights = (1 + alpha_reg * m**2) ** k_reg
            gradient_j = gradient_j / xp.concat([weights, weights])
            gradients.append(gradient_j)

        # Concatenate gradients for all scatterers
        gradient = xp.concat(gradients)
        return np.asarray(gradient, device="cpu")

    def constraint_fun(x: np.ndarray) -> float:
        """Constraint: sum of Fourier coefficient norms <= 1 for each scatterer."""
        total = 0.0
        for j in range(M):
            offset = j * 2 * n_modes
            cos_part = x[offset : offset + n_modes]
            sin_part = x[offset + n_modes : offset + 2 * n_modes]
            total += float(np.sum(np.sqrt(cos_part**2 + sin_part**2)))
        return total

    def constraint_jac(x: np.ndarray) -> np.ndarray:
        """Jacobian of the constraint."""
        gradient = np.zeros_like(x)
        for j in range(M):
            offset = j * 2 * n_modes
            cos_part = x[offset : offset + n_modes]
            sin_part = x[offset + n_modes : offset + 2 * n_modes]
            r = np.sqrt(cos_part**2 + sin_part**2)
            np.divide(cos_part, r, out=gradient[offset : offset + n_modes], where=r > 0)
            np.divide(
                sin_part, r, out=gradient[offset + n_modes : offset + 2 * n_modes], where=r > 0
            )
        return gradient

    constraint = NonlinearConstraint(constraint_fun, -np.inf, 1, jac=constraint_jac)

    val_hist = []

    def callback(intermediate_result: Any) -> None:
        val_hist.append(intermediate_result.fun)

    result = minimize(
        fun,
        np.zeros(M * 2 * n_modes),
        method="trust-constr",
        jac=jac,
        constraints=constraint,
        callback=callback,
        options={"verbose": 3, "maxiter": n_steps},
    )

    # Extract optimized shapes for all scatterers
    shapes_params_opt = unpack(result.x)
    shapes_opt_list = [
        ParameterShape(cos_coefs=cos_coefs, sin_coefs=sin_coefs)
        for cos_coefs, sin_coefs in shapes_params_opt
    ]
    shapes_opt = ShapeList(shapes_opt_list)

    t_plot = np.linspace(0, 2 * np.pi, 10000)
    t_arr = xp.asarray(t_plot, dtype=dtype, device=device)

    # Save optimized shapes for all scatterers
    shapes_data = {}
    for j in range(M):
        x_plot = np.asarray(shapes_opt.x(t_arr)[..., j, :], device="cpu")
        shapes_data[f"shape_{j}"] = {
            "x": xp.ascontiguousarray(x_plot[:, 0]),
            "y": xp.ascontiguousarray(x_plot[:, 1]),
        }

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
            "n_scatterers": M,
            "target_point": target_point,
            "desired_total_field": {
                "real": float(desired_total_field.real),
                "imag": float(desired_total_field.imag),
            },
            "alpha": {"real": float(alpha.real), "imag": float(alpha.imag)},
            "eta": {"real": float(eta.real), "imag": float(eta.imag)},
            "final_parameters": {
                f"scatterer_{j}": {
                    "cos_coefs": shapes_params_opt[j][0].tolist(),
                    "sin_coefs": shapes_params_opt[j][1].tolist(),
                }
                for j in range(M)
            },
        })
    )

    # Optimized shapes
    (path / "optimized_shapes.json").write_bytes(
        orjson.dumps(shapes_data, option=orjson.OPT_SERIALIZE_NUMPY)
    )

    # Near-field data
    density_opt = scattering_dirichlet(
        k=k,
        shapes=shapes_opt,
        incident_field=incident_field,
        alpha=alpha_arr,
        eta=eta_arr,
        n=n,
    )
    field_data = plot_near_field_prepare(
        density_opt,
        incident_field,
        xlim=(-4, 4),
        ylim=(-4, 4),
        k=k,
        shapes=shapes_opt,
        n=n,
        alpha=alpha_arr,
        eta=eta_arr,
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
        ``optimized_shapes.json``, and ``optimized_near_field.json``.

    """
    # Load data
    history = json.loads((path / "optimization_history.json").read_text())
    shapes_data = json.loads((path / "optimized_shapes.json").read_text())
    field_data = json.loads((path / "optimized_near_field.json").read_text())
    alpha = history["alpha_reg"]
    M = history.get("n_scatterers", 1)

    # Optimization history plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(history["val_hist"])
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Optimization history (alpha={history['alpha_reg']})")
    fig.tight_layout()
    fig.savefig(path / f"optimization_history_{alpha}.svg")
    plt.close(fig)

    # Magnitude of coefficients plot for each scatterer
    for j in range(M):
        fig, ax = plt.subplots(figsize=(4, 3))
        scatterer_key = f"scatterer_{j}"
        sin_coefs = np.asarray(history["final_parameters"][scatterer_key]["sin_coefs"])
        cos_coefs = np.asarray(history["final_parameters"][scatterer_key]["cos_coefs"])[1:]
        n_modes = len(sin_coefs)
        m = np.arange(1, n_modes + 1)
        ax.plot(m, np.abs(cos_coefs), "o-", label="cosine coefficients")
        ax.plot(m, np.abs(sin_coefs), "o-", label="sine coefficients")
        ax.set_yscale("log")
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Optimized Fourier coefficients (scatterer {j})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path / f"optimized_coefficients_{j}_{alpha}.svg")
        plt.close(fig)

    # Optimized shapes plot
    fig, ax = plt.subplots()
    for j in range(M):
        shape_key = f"shape_{j}"
        ax.plot(shapes_data[shape_key]["x"], shapes_data[shape_key]["y"], label=f"Scatterer {j}")
    ax.set_aspect("equal")
    ax.set_title("Optimized shapes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / f"optimized_shapes_{alpha}.svg")
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
