from __future__ import annotations

import json
import math
import pathlib
from pathlib import Path
from typing import Any

import array_api_compat.numpy as np_compat
import orjson
from array_api.latest import Array
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from ie_circle import NystromInterpolant, trapezoidal_quadrature
from matplotlib import pyplot as plt
from tqdm import tqdm

from biem_helmholtz_2d._acoustic import (
    near_field,
    plot_near_field_prepare,
    scattering_dirichlet,
)
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import field_objective_derivative
from biem_helmholtz_2d._pairs import off_diagonal_indices
from biem_helmholtz_2d._potential_inner import kernel as near_kernel
from biem_helmholtz_2d._potential_inner_derivative import kernel as near_derivative_kernel
from biem_helmholtz_2d.optimization._shape import ParameterShape


def project_coefficients(coefficients: Array, /, *, weights: Array | None = None) -> Array:
    r"""
    Project Fourier pairs onto $\sum_m\sqrt{a_m^2+b_m^2}\leq1$.

    Parameters
    ----------
    coefficients : Array
        Real cosine and sine coefficients of shape (..., n_modes, 2).
    weights : Array or None
        Positive metric weights of shape (n_modes,). None selects the
        Euclidean projection from the constrained optimization theorem.

    Returns
    -------
    Array
        Metric projection with the same shape. For weights $w_m$, the
        projected radii are $\max(r_m-\lambda/w_m,0)$.

    Examples
    --------
    >>> import array_api_compat.numpy as xp
    >>> project_coefficients(xp.asarray([[3., 0.], [0., 1.]]))
    array([[1., 0.],
           [0., 0.]])

    """
    info = check_shapes("...MC,C", coefficients, (2,), names="coefficients,components")
    # A singleton component axis must not broadcast to a cosine/sine pair.
    if coefficients.shape[-1] != 2:
        raise ValueError("coefficients must contain cosine/sine pairs")
    n_modes = info.unique["M"].shape_broadcasted[0]
    xp = array_namespace(coefficients, weights)
    if not xp.isdtype(coefficients.dtype, "real floating"):
        raise ValueError("coefficients must have a real floating dtype")
    if weights is None:
        weights = xp.ones(n_modes, device=coefficients.device, dtype=coefficients.dtype)
    check_shapes("...MC,M,C", coefficients, weights, (2,), names="coefficients,weights,components")
    weights = xp.broadcast_to(weights, (n_modes,))
    if not bool(xp.all(xp.isfinite(weights) & (weights > 0))):
        raise ValueError("weights must be positive and finite")
    if n_modes == 0:
        return xp.copy(coefficients)
    radii = xp.linalg.vector_norm(coefficients, axis=-1)
    weights = xp.broadcast_to(weights, radii.shape)
    order = xp.argsort(weights * radii, axis=-1, descending=True)
    ordered_radii = xp.take_along_axis(radii, order, axis=-1)
    ordered_weights = xp.take_along_axis(weights, order, axis=-1)
    thresholds = (xp.cumulative_sum(ordered_radii, axis=-1) - 1) / xp.cumulative_sum(
        1 / ordered_weights, axis=-1
    )
    active = ordered_weights * ordered_radii > thresholds
    threshold = xp.clip(xp.max(xp.where(active, thresholds, -xp.inf), axis=-1), min=0)
    scale = xp.clip(radii - threshold[..., None] / weights, min=0) / xp.where(radii > 0, radii, 1)
    return xp.where(
        (xp.sum(radii, axis=-1) <= 1)[..., None, None],
        coefficients,
        coefficients * scale[..., None],
    )


def example_optimization(
    *,
    path: Path,
    centers: Array,
    k: complex,
    alpha: complex,
    eta: complex,
    n: int,
    n_quad: int | None = None,
    n_steps: int = 10,
    n_modes: int = 20,
    alpha_reg: float = 0.1,
    k_reg: int = 3,
    desired_total_field: complex = 0j,
    target_point: tuple[float, float] = (-2, 3),
) -> None:
    r"""
    Example optimization using regularized projected gradient descent.

    Minimizes $|u_{\mathrm{total}}(x_0) - c|^2$ subject to
    $\sum \sqrt{a_n^2 + b_n^2} \le 1$, where
    $u_{\mathrm{total}} = u_{\mathrm{scat}} + u_{\mathrm{inc}}$.
    The constant Fourier coefficient is fixed to $1$ on each body. The
    coefficients of each body are projected independently. Centers are fixed
    and separated by more than $4$. At equality the constraint can admit a
    zero radius, so it does not by itself guarantee a regular boundary.

    Saves plot-ready data as JSON files in ``path``. Use
    :func:`example_optimization_plot` to generate figures.

    Parameters
    ----------
    path : Path
        Directory to save JSON files.
    n : int
        Nyström parameter; the forward and adjoint solves use $2n-1$ nodes.
    n_quad : int or None
        Integration parameter for field and sensitivity, independent of the solve.
        None uses ``n``; the number of integration nodes is $2n_{\rm quad}-1$.
    centers : Array
        Fixed scatterer centers of shape (n_scatterers, 2), pairwise separated
        by more than 4. The target must be more than 2 from every center.
    n_modes : int
        Number of nonconstant Fourier modes.
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
    xp = array_namespace(centers)
    if n_quad is None:
        n_quad = n
    if n < 1 or n_quad < 1:
        raise ValueError("n and n_quad must be positive")
    dtype, device = centers.dtype, centers.device
    if not xp.isdtype(dtype, "real floating"):
        raise ValueError("centers must have a real floating dtype")
    info = check_shapes("CD,D", centers, (2,), names="centers,coordinate_dimension")
    if centers.shape[-1] != 2:
        raise ValueError("centers must contain two coordinates")
    n_scatterers = info.unique["C"].shape_broadcasted[0]
    if n_scatterers == 0:
        raise ValueError("at least one center is required")
    if not bool(xp.all(xp.isfinite(centers))):
        raise ValueError("centers must be finite")
    rows, columns = off_diagonal_indices(n_scatterers, xp=xp, device=device)
    separation = xp.linalg.vector_norm(
        xp.take(centers, rows, axis=0) - xp.take(centers, columns, axis=0), axis=-1
    )
    if bool(xp.any(separation <= 4)):
        raise ValueError("centers must be separated by more than 4")
    path.mkdir(parents=True, exist_ok=True)
    k = xp.asarray(k, device=device, dtype=xp.result_type(dtype, k))
    eta = xp.asarray(eta, device=device, dtype=xp.result_type(dtype, eta))
    alpha = xp.asarray(alpha, device=device, dtype=xp.result_type(dtype, alpha))
    point = xp.asarray(target_point, device=device, dtype=dtype)
    check_shapes("D,D", point, (2,), names="target_point,coordinate_dimension")
    if point.shape[-1] != 2:
        raise ValueError("target_point must contain two coordinates")
    if not bool(xp.all(xp.isfinite(point))):
        raise ValueError("target_point must contain finite coordinates")
    if bool(xp.any(xp.linalg.vector_norm(centers - point, axis=-1) <= 2)):
        raise ValueError("target_point must be more than 2 from every center")
    direction = xp.asarray([1, 0], device=device, dtype=dtype)
    incident_field = plane_wave(k, direction)
    incident_field_grad = plane_wave_grad(k, direction)

    # Each perturbation batch varies the same Fourier coefficient on each boundary.
    eye_modes = xp.eye(n_modes, dtype=dtype, device=device)
    eye_bodies = xp.eye(n_scatterers, dtype=dtype, device=device)
    zeros = xp.zeros((n_scatterers, n_scatterers, n_modes), dtype=dtype, device=device)
    zero_constant = xp.zeros((n_scatterers, n_scatterers, 1), dtype=dtype, device=device)
    zero_centers = xp.zeros((n_scatterers, n_scatterers, 2), dtype=dtype, device=device)
    basis = [
        (
            ParameterShape(
                cos_coefs=xp.concat([zero_constant, eye_bodies[..., None] * eye_modes[m]], axis=-1),
                sin_coefs=zeros,
                center=zero_centers,
            ),
            ParameterShape(
                cos_coefs=xp.concat([zero_constant, zeros], axis=-1),
                sin_coefs=eye_bodies[..., None] * eye_modes[m],
                center=zero_centers,
            ),
        )
        for m in range(n_modes)
    ]

    def unpack(x: Array, /) -> tuple[Array, Array]:
        cos = xp.concat(
            [xp.ones((n_scatterers, 1), dtype=dtype, device=device), x[..., 0]], axis=-1
        )
        return cos, x[..., 1]

    def make_shapes(x: Array, /) -> ParameterShape:
        cos, sin = unpack(x)
        return ParameterShape(cos_coefs=cos, sin_coefs=sin, center=centers)

    def solve(x: Array, /) -> tuple[ParameterShape, NystromInterpolant, Array]:
        shapes = make_shapes(x)
        phi = scattering_dirichlet(
            k=k, shape=shapes, incident_field=incident_field, alpha=alpha, eta=eta, n=n
        )
        field = near_field(phi, point, k=k, shape=shapes, n=n_quad, alpha=alpha, eta=eta)
        return shapes, phi, field

    target = desired_total_field - incident_field(point)

    def fun(x: Array, /) -> float:
        _, _, field = solve(x)
        return float(xp.abs(field - target) ** 2)

    def jac(x: Array, /) -> Array:
        if n_modes == 0:
            return xp.zeros_like(x)
        shapes, phi, field = solve(x)
        t, _ = trapezoidal_quadrature(
            n_quad,
            xp=xp,
            device=device,
            dtype=xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta)),
        )
        _, directional = field_objective_derivative(
            phi,
            point[None, :],
            (2 * (field - target))[None],
            shape=shapes,
            incident_gradient=incident_field_grad(shapes.x(t)),
            k=k,
            alpha=alpha,
            eta=eta,
            n_solve=n,
            n_quad=n_quad,
            kernel=near_kernel,
            derivative_kernel=near_derivative_kernel,
        )

        return xp.moveaxis(
            xp.stack([xp.stack([directional(h) for h in pair]) for pair in basis]), -1, 0
        )

    # The Euclidean derivative remains distinct from the Hilbert Riesz gradient.
    modes = xp.arange(1, n_modes + 1, dtype=dtype, device=device)
    weights = (1 + alpha_reg * modes**2) ** k_reg
    if alpha_reg < 0 or k_reg < 0 or not bool(xp.all(xp.isfinite(weights))):
        raise ValueError("alpha_reg and k_reg must define finite positive Sobolev weights")

    x = xp.zeros((n_scatterers, n_modes, 2), dtype=dtype, device=device)
    pbar = tqdm(range(n_steps), desc=f"alpha_reg={alpha_reg:g}", unit="step")
    pbar.set_postfix_str("initial solve")
    value = fun(x)
    val_hist = [value]
    for _ in pbar:
        pbar.set_postfix_str(f"objective={value:.3e}, gradient")
        derivative = jac(x)
        riesz = derivative / weights[:, None]
        norm = xp.sqrt(xp.sum(weights[:, None] * riesz**2))
        if not bool(xp.isfinite(norm)) or not bool(norm > 0):
            break
        # Project in the same Hilbert metric as the Riesz direction. This
        # preserves descent even at an active constraint.
        direction = riesz / norm
        step = 1.0
        while step >= xp.finfo(dtype).eps:
            pbar.set_postfix_str(f"objective={value:.3e}, line search={step:.2g}")
            candidate = project_coefficients(x - step * direction, weights=weights)
            slope = float(xp.sum(derivative * (candidate - x)))
            if slope >= 0:
                break
            trial = fun(candidate)
            if math.isfinite(trial) and trial <= value + 1e-4 * slope:
                x, value = candidate, trial
                val_hist.append(value)
                break
            step /= 2
        else:
            break
        if slope >= 0:
            break

    cos_coefs_opt, sin_coefs_opt = unpack(x)
    shapes_opt = make_shapes(x)

    t_plot = xp.linspace(0, 2 * xp.pi, 10000, dtype=dtype, device=device)
    x_plot = xp.asarray(xp.moveaxis(shapes_opt.x(t_plot), -2, 0), device="cpu")
    centers_cpu = xp.asarray(centers, device="cpu")

    def real_array_to_list(values: Array, /) -> Any:
        return np_compat.asarray(xp.asarray(values, device="cpu")).tolist()

    # Optimization history
    k_value, alpha_value, eta_value = complex(k), complex(alpha), complex(eta)
    (path / "optimization_history.json").write_bytes(
        orjson.dumps({
            "val_hist": val_hist,
            "k": {"real": k_value.real, "imag": k_value.imag},
            "n": n,
            "n_quad": n_quad,
            "alpha_reg": alpha_reg,
            "k_reg": k_reg,
            "n_modes": n_modes,
            "n_steps": n_steps,
            "target_point": target_point,
            "centers": real_array_to_list(centers_cpu),
            "desired_total_field": {
                "real": float(desired_total_field.real),
                "imag": float(desired_total_field.imag),
            },
            "alpha": {"real": alpha_value.real, "imag": alpha_value.imag},
            "eta": {"real": eta_value.real, "imag": eta_value.imag},
            "final_parameters": {
                "cos_coefs": real_array_to_list(xp.asarray(cos_coefs_opt, device="cpu")),
                "sin_coefs": real_array_to_list(xp.asarray(sin_coefs_opt, device="cpu")),
            },
        })
    )

    # Optimized shape
    (path / "optimized_shape.json").write_bytes(
        orjson.dumps({
            "x": real_array_to_list(x_plot[..., 0]),
            "y": real_array_to_list(x_plot[..., 1]),
        })
    )

    # Near-field data
    density_opt = scattering_dirichlet(
        k=k, shape=shapes_opt, incident_field=incident_field, alpha=alpha, eta=eta, n=n
    )
    field_data = plot_near_field_prepare(
        density_opt,
        incident_field,
        xlim=(
            min(float(xp.min(centers_cpu[:, 0])) - 2.5, target_point[0] - 1),
            max(float(xp.max(centers_cpu[:, 0])) + 2.5, target_point[0] + 1),
        ),
        ylim=(
            min(float(xp.min(centers_cpu[:, 1])) - 3, target_point[1] - 1),
            max(float(xp.max(centers_cpu[:, 1])) + 3, target_point[1] + 1),
        ),
        k=k,
        shape=shapes_opt,
        n=n_quad,
        alpha=alpha,
        eta=eta,
        n_plot=200,
        isin_shape_n_quad=500,
        isin_shape_tol=1e-5,
    )
    field_json = {
        field: {
            component: {
                **entry,
                "data": real_array_to_list(xp.asarray(entry["data"], device="cpu")),
            }
            for component, entry in components.items()
        }
        for field, components in field_data.items()
    }
    (path / "optimized_near_field.json").write_bytes(orjson.dumps(field_json))


def example_optimization_plot(path: pathlib.Path, /) -> None:
    """Compare optimized shapes, coefficients, and histories for one regularization value."""
    files = list(path.glob("n_modes_*/optimization_history.json"))
    if not files:
        files = [path / "optimization_history.json"]
    results = [
        (
            json.loads(file.read_text()),
            json.loads((file.parent / "optimized_shape.json").read_text()),
        )
        for file in files
    ]
    results.sort(key=lambda result: result[0]["n_modes"])
    alpha = results[0][0]["alpha_reg"]
    if any(history["alpha_reg"] != alpha for history, _ in results):
        raise ValueError("All runs must have the same alpha_reg")
    extent = json.loads((files[0].parent / "optimized_near_field.json").read_text())["utot"]["re"][
        "extent"
    ]

    fig, axes = plt.subplots(
        1, len(results), figsize=(4 * len(results), 3), sharex=True, sharey=True, squeeze=False
    )
    for ax, (history, shape) in zip(axes[0], results, strict=True):
        xs, ys = shape["x"], shape["y"]
        if xs and isinstance(xs[0], (int, float)):
            xs, ys = [xs], [ys]
        for x, y in zip(xs, ys, strict=True):
            ax.plot(x, y, color="black")
        if "target_point" in history:
            ax.plot(
                *history["target_point"],
                "X",
                markersize=4 * plt.rcParams["lines.markersize"],
                markerfacecolor="black",
                markeredgewidth=2,
                markeredgecolor="white",
            )
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x", title=f"{history['n_modes']} modes")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("y")
    fig.suptitle(f"Optimized shapes (alpha={alpha})")
    fig.tight_layout()
    fig.savefig(path / f"optimized_shape_{alpha}.svg")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(results), figsize=(4 * len(results), 3), sharex=True, sharey=True, squeeze=False
    )
    for ax, (history, _) in zip(axes[0], results, strict=True):
        cos = np_compat.asarray(history["final_parameters"]["cos_coefs"])
        sin = np_compat.asarray(history["final_parameters"]["sin_coefs"])
        if cos.ndim == 1:
            cos, sin = cos[None, :], sin[None, :]
        modes = range(1, sin.shape[1] + 1)
        for i in range(cos.shape[0]):
            ax.semilogy(
                modes, np_compat.abs(cos[i, 1:]), "o-", color=f"C{i}", label=f"body {i + 1} cos"
            )
            ax.semilogy(
                modes, np_compat.abs(sin[i]), "o--", color=f"C{i}", label=f"body {i + 1} sin"
            )
        ax.set(xlabel="Mode number", title=f"{history['n_modes']} modes")
    axes[0, 0].set_ylabel("Coefficient magnitude")
    axes[0, 0].legend()
    fig.suptitle(f"Optimized Fourier coefficients (alpha={alpha})")
    fig.tight_layout()
    fig.savefig(path / f"optimized_coefficients_{alpha}.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for history, _ in results:
        ax.semilogy(history["val_hist"], label=f"{history['n_modes']} modes")
    ax.set(
        xlabel="Iteration", ylabel="Objective value", title=f"Optimization history (alpha={alpha})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / f"optimization_history_{alpha}.svg")
    plt.close(fig)
