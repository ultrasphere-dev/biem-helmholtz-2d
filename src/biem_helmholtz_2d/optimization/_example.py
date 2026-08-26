from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import orjson
from array_api.latest import Array, ArrayNamespace
from ie_circle import NystromInterpolant, trapezoidal_quadrature
from matplotlib import pyplot as plt
from scipy.optimize import NonlinearConstraint, minimize

from biem_helmholtz_2d._acoustic import (
    FieldData,
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
    n_modes: int = 20,
    alpha_reg: float = 0.1,
    k_reg: int = 3,
    desired_total_field: complex = 0j,
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
    xp : ArrayNamespace
        Array API namespace.
    dtype : Any
        Array dtype.
    device : Any
        Array device.
    n_modes : int
        Number of Fourier modes to optimize.
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

    """
    path.mkdir(parents=True, exist_ok=True)
    n = n_modes + 20
    k = xp.asarray(2.0, device=device, dtype=dtype)
    eta = xp.asarray(0.0, device=device, dtype=dtype)
    alpha = xp.asarray(1.0, device=device, dtype=dtype)
    point = xp.asarray([-2.0, 3.0], device=device, dtype=dtype)
    direction = xp.asarray([1.0, 0.0], device=device, dtype=dtype)
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
        dr_j = 2.0 * xp.real(
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

    constraint = NonlinearConstraint(constraint_fun, -np.inf, 1.0, jac=constraint_jac)

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
        options={"verbose": 1, "maxiter": 100},
    )

    cos_coefs_opt, sin_coefs_opt = unpack(result.x)
    shape_opt = ParameterShape(cos_coefs=cos_coefs_opt, sin_coefs=sin_coefs_opt)

    # --- Prepare plot-ready data and save as JSON ---

    t_plot = np.linspace(0, 2 * np.pi, 10000)
    t_arr = xp.asarray(t_plot, dtype=dtype, device=device)
    x_plot = np.asarray(shape_opt.x(t_arr), device="cpu")

    # Optimization history
    (path / "optimization_history.json").write_bytes(
        orjson.dumps({
            "val_hist": val_hist,
            "k": float(k),
            "n": n,
            "alpha_reg": alpha_reg,
            "k_reg": k_reg,
        })
    )

    # Optimized shape
    (path / "optimized_shape.json").write_bytes(
        orjson.dumps({
            "x": x_plot[:, 0],
            "y": x_plot[:, 1],
        })
    )

    # Near-field data
    density_opt = scattering_dirichlet(
        k=k, shape=shape_opt, incident_field=incident_field, alpha=alpha, eta=eta, n=n
    )
    field_data = plot_near_field_prepare(
        density_opt,
        incident_field,
        xlim=(-4.0, 4.0),
        ylim=(-4.0, 4.0),
        k=k,
        shape=shape_opt,
        n=n,
        alpha=alpha,
        eta=eta,
        n_plot=200,
        isin_shape_n_quad=500,
        isin_shape_tol=1e-5,
    )
    field_serializable: dict[str, dict[str, Any]] = {}
    for (field_name, component), entry in field_data.items():
        key = f"{field_name}_{component}"
        field_serializable[key] = {
            "data": np.asarray(entry["data"]),
            "vmax": entry["vmax"],
            "vmin": entry["vmin"],
            "extent": list(entry["extent"]),
        }
    (path / "optimized_near_field.json").write_bytes(
        orjson.dumps(
            {
                "field": field_serializable,
                "point_x": float(point[0]),
                "point_y": float(point[1]),
            },
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
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

    # Optimization history plot
    fig, ax = plt.subplots()
    ax.plot(history["val_hist"])
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(
        f"Optimization history "
        f"(k={history['k']}, n={history['n']}, "
        f"alpha_reg={history['alpha_reg']}, k_reg={history['k_reg']})"
    )
    fig.tight_layout()
    fig.savefig(path / "optimization_history.png")
    plt.close(fig)

    # Optimized shape plot
    fig, ax = plt.subplots()
    ax.plot(shape_data["x"], shape_data["y"])
    ax.set_aspect("equal")
    ax.set_title("Optimized shape")
    fig.tight_layout()
    fig.savefig(path / "optimized_shape.png")
    plt.close(fig)

    # Near-field plot
    field_serializable = field_data["field"]
    point_x = field_data["point_x"]
    point_y = field_data["point_y"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_near_field(
        _reconstruct_field_data(field_serializable),
        ax_utot_re=ax[0],
        ax_utot_im=ax[1],
        ax_utot_abs=ax[2],
    )
    for a in ax:
        a.plot(
            point_x,
            point_y,
            "X",
            markersize=25,
            markerfacecolor="black",
            markeredgewidth=3,
            markeredgecolor="white",
            label="Point to minimize",
        )
        a.legend()
    fig.tight_layout()
    fig.savefig(path / "optimized_near_field.png")
    plt.close(fig)


def _reconstruct_field_data(
    field_serializable: dict[str, dict[str, Any]],
) -> FieldData:
    """Reconstruct :class:`FieldData` from its JSON-serializable representation."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, entry in field_serializable.items():
        field_name, component = key.split("_", 1)
        result[field_name, component] = {
            "data": np.array(entry["data"]),
            "vmax": entry["vmax"],
            "vmin": entry["vmin"],
            "extent": tuple(entry["extent"]),
        }
    return cast(FieldData, result)
