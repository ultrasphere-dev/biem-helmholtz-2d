from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable
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

from biem_helmholtz_2d._acoustic import plot_near_field_prepare, scattering_dirichlet
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import field_objective, field_objective_derivative
from biem_helmholtz_2d._pairs import off_diagonal_indices
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


def _two_way_backtracking(
    x: Array,
    /,
    *,
    fun: Callable[[Array], float],
    value: float,
    derivative: Array,
    direction: Array,
    weights: Array,
    step: float,
    tau: float,
    c: float,
) -> tuple[Array, float, float] | None:
    r"""
    Find a projected Armijo step using two-way backtracking.

    A trial $y=P_C^W(x-sd)$ is accepted when
    $f(y)\leq f(x)+c\,\nabla f(x)\cdot(y-x)$. The search expands an
    accepted step or contracts a rejected step, returning the best accepted
    trial. Expansion is capped at step size 1.

    Parameters
    ----------
    x : Array
        Current feasible coefficients of shape (..., M, 2).
    fun : Callable[[Array], float]
        Objective evaluated at coefficients of shape (..., M, 2).
    value : float
        Objective value at ``x``.
    derivative : Array
        Euclidean objective derivative of shape (..., M, 2).
    direction : Array
        Descent direction of shape (..., M, 2), subtracted from ``x``.
    weights : Array
        Positive projection metric weights of shape (M,).
    step : float
        Positive initial trial step, normally the previously accepted step.
    tau : float
        Contraction factor in $(0, 1)$; expansion divides by this factor.
    c : float
        Armijo sufficient-decrease coefficient in $(0, 1)$.

    Returns
    -------
    tuple[Array, float, float] or None
        Accepted coefficients, objective value and step, or None if no descent
        can be verified for the projected step.

    Raises
    ------
    ValueError
        If ``tau`` or ``c`` is outside $(0, 1)$.
    FloatingPointError
        If the projected slope is not finite.
    RuntimeError
        If the projected slope is materially positive or the search fails.

    Warns
    -----
    RuntimeWarning
        If the projected slope is zero or nonnegative within roundoff.

    Examples
    --------
    >>> _two_way_backtracking(
    ...     np_compat.asarray([[0., 0.]]),
    ...     fun=lambda y: float(np_compat.sum((y - np_compat.asarray([[0.2, 0.]])) ** 2) / 2),
    ...     value=0.02, derivative=np_compat.asarray([[-0.2, 0.]]),
    ...     direction=np_compat.asarray([[-1., 0.]]), weights=np_compat.ones(1),
    ...     step=0.1, tau=0.5, c=1e-4,
    ... )[2]
    0.2

    """
    if not 0 < tau < 1:
        raise ValueError("tau must be between 0 and 1")
    if not 0 < c < 1:
        raise ValueError("c must be between 0 and 1")
    xp = array_namespace(x, derivative, direction, weights)
    best = None
    shrinking = False
    while step >= xp.finfo(x.dtype).eps:
        candidate = project_coefficients(x - step * direction, weights=weights)
        slope = float(xp.sum(derivative * (candidate - x)))
        if not math.isfinite(slope):
            raise FloatingPointError(f"Projected slope is not finite: {slope}")
        if slope >= 0:
            # Projection can round a stationary step by a few ulps.
            scale = xp.sum(xp.abs(derivative) * (xp.abs(candidate) + xp.abs(x)))
            roundoff = 8 * xp.finfo(x.dtype).eps * float(scale)
            if slope > roundoff:
                raise RuntimeError(f"Projected slope is positive: {slope}")
            warnings.warn(
                f"Projected slope is nonnegative ({slope}); no descent can be verified.",
                RuntimeWarning,
                stacklevel=2,
            )
            return best
        trial = fun(candidate)
        if math.isfinite(trial) and trial <= value + c * slope:
            best = candidate, trial, step
            if shrinking or step >= 1:
                break
            step = min(step / tau, 1)
        else:
            if best is not None:
                break
            shrinking = True
            step *= tau
    if best is None:
        raise RuntimeError("Armijo line search failed to find a decreasing step")
    return best


def example_optimization(
    *,
    path: Path,
    centers: Array,
    target: Array,
    weight_target: float,
    include_cross_section: bool,
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
) -> None:
    r"""
    Example optimization using regularized projected gradient descent.

    Minimizes the mean squared total-field error at ``target``, optionally
    adding the scattering cross section, subject to
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
        Desired total field value $c$ at each target.
    k : complex
        The wave number.
    alpha : complex
        The coupling parameter for the double-layer potential.
    eta : complex
        The coupling parameter for the single-layer potential.
    target : Array
        Observation points of shape (n_targets, 2).
    weight_target : float
        Weight of the mean squared point-field error.
    include_cross_section : bool
        Include the scattering cross section in the objective.

    """
    xp = array_namespace(centers, target)
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
    if target.ndim != 2 or target.shape[0] == 0 or target.shape[1] != 2:
        raise ValueError("target must have shape (n_targets, 2) with n_targets > 0")
    if not xp.isdtype(target.dtype, "real floating") or not bool(xp.all(xp.isfinite(target))):
        raise ValueError("target must contain finite real floating-point coordinates")
    if bool(xp.any(xp.linalg.vector_norm(centers[None, :, :] - target[:, None, :], axis=-1) <= 2)):
        raise ValueError("each target must be more than 2 from every center")
    if not math.isfinite(weight_target) or weight_target < 0:
        raise ValueError("weight_target must be finite and nonnegative")
    if include_cross_section and desired_total_field != 0:
        raise ValueError("the cloaking objective minimizes the total field itself")
    path.mkdir(parents=True, exist_ok=True)
    k = xp.asarray(k, device=device, dtype=xp.result_type(dtype, k))
    eta = xp.asarray(eta, device=device, dtype=xp.result_type(dtype, eta))
    alpha = xp.asarray(alpha, device=device, dtype=xp.result_type(dtype, alpha))
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

    def evaluate(x: Array, /) -> tuple[ParameterShape, NystromInterpolant, Array, Any]:
        shapes = make_shapes(x)
        phi = scattering_dirichlet(
            k=k, shape=shapes, incident_field=incident_field, alpha=alpha, eta=eta, n=n
        )
        value, terms = field_objective(
            phi,
            target=target,
            weight_target=weight_target,
            include_cross_section=include_cross_section,
            shape=shapes,
            incident_field=incident_field,
            k=k,
            alpha=alpha,
            eta=eta,
            n_quad=n_quad,
            desired_total_field=desired_total_field,
        )
        return shapes, phi, value, terms

    def fun(x: Array, /) -> float:
        return float(evaluate(x)[2])

    def jac(x: Array, /) -> Array:
        if n_modes == 0:
            return xp.zeros_like(x)
        shapes, phi, _, terms = evaluate(x)
        t, _ = trapezoidal_quadrature(
            n_quad,
            xp=xp,
            device=device,
            dtype=xp.result_type(xp.real(k), xp.real(alpha), xp.real(eta)),
        )
        _, directional = field_objective_derivative(
            phi,
            terms,
            shape=shapes,
            incident_gradient=incident_field_grad(shapes.x(t)),
            k=k,
            alpha=alpha,
            eta=eta,
            n_solve=n,
            n_quad=n_quad,
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
    step = 1.0
    for _ in pbar:
        pbar.set_postfix_str(f"objective={value:.3e}, gradient")
        derivative = jac(x)
        riesz = derivative / weights[:, None]
        norm = xp.sqrt(xp.sum(weights[:, None] * riesz**2))
        if not bool(xp.isfinite(norm)):
            raise FloatingPointError("Optimization gradient norm is not finite")
        if not bool(norm > 0):
            warnings.warn(
                "Optimization stopped: gradient norm is zero.", RuntimeWarning, stacklevel=2
            )
            break
        # Project in the same Hilbert metric as the Riesz direction. This
        # preserves descent even at an active constraint.
        direction = riesz / norm
        pbar.set_postfix_str(f"objective={value:.3e}, line search")
        accepted = _two_way_backtracking(
            x,
            fun=fun,
            value=value,
            derivative=derivative,
            direction=direction,
            weights=weights,
            step=step,
            tau=0.5,
            c=1e-4,
        )
        if accepted is None:
            break
        x, value, step = accepted
        val_hist.append(value)
        pbar.set_postfix_str(f"objective={value:.3e}, step={step:.2g}")

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
            "target": real_array_to_list(target),
            "weight_target": weight_target,
            "include_cross_section": include_cross_section,
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
            min(float(xp.min(centers_cpu[:, 0])) - 2.5, float(xp.min(target[:, 0])) - 1),
            max(float(xp.max(centers_cpu[:, 0])) + 2.5, float(xp.max(target[:, 0])) + 1),
        ),
        ylim=(
            min(float(xp.min(centers_cpu[:, 1])) - 3, float(xp.min(target[:, 1])) - 1),
            max(float(xp.max(centers_cpu[:, 1])) + 3, float(xp.max(target[:, 1])) + 1),
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


def example_optimization_plot(path: Path, /) -> None:
    """Compare regularization values and mode counts in three study-level figures."""
    files = sorted([
        *path.glob("alpha_reg_*/n_modes_*/optimization_history.json"),
        *path.glob("alpha_reg_*/optimization_history.json"),
        *path.glob("n_modes_*/optimization_history.json"),
        *(
            [path / "optimization_history.json"]
            if (path / "optimization_history.json").is_file()
            else []
        ),
    ])
    if not files:
        raise ValueError(f"No optimization data in {path}")
    results = [
        (
            json.loads(file.read_text()),
            json.loads((file.parent / "optimized_shape.json").read_text()),
        )
        for file in files
    ]
    alphas = sorted({history["alpha_reg"] for history, _ in results})
    modes = sorted({history["n_modes"] for history, _ in results})
    colors = {mode: f"C{i % 10}" for i, mode in enumerate(modes)}
    handles = [plt.Line2D([], [], color=colors[mode], label=f"{mode} modes") for mode in modes]
    extent = json.loads((files[0].parent / "optimized_near_field.json").read_text())["utot"]["re"][
        "extent"
    ]
    count = len(alphas)

    fig, axes = plt.subplots(
        1, count, figsize=(4 * count, 4.5), sharex=True, sharey=True, squeeze=False
    )
    for column, alpha in enumerate(alphas):
        ax = axes[0, column]
        for history, shape in results:
            if history["alpha_reg"] != alpha:
                continue
            xs, ys = shape["x"], shape["y"]
            if xs and isinstance(xs[0], (int, float)):
                xs, ys = [xs], [ys]
            for x, y in zip(xs, ys, strict=True):
                ax.plot(x, y, color=colors[history["n_modes"]])
        history = next(history for history, _ in results if history["alpha_reg"] == alpha)
        targets = history.get("target")
        if targets is None:
            targets = [history["target_point"]] if history.get("target_point") is not None else []
        if len(targets) > 1:
            points = np_compat.asarray(targets)
            ax.plot(points[:, 0], points[:, 1], ".", color="black", markersize=3)
        elif targets:
            ax.plot(
                *targets[0],
                "X",
                markersize=4 * plt.rcParams["lines.markersize"],
                markerfacecolor="black",
                markeredgewidth=2,
                markeredgecolor="white",
            )
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x", title=f"alpha={alpha:g}")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("y")
    fig.suptitle("Optimized shapes")
    fig.legend(handles=handles, loc="lower center", ncols=len(handles))
    fig.tight_layout(rect=(0, 0.18, 1, 0.95))
    destination = path / "optimized_shape.svg"
    if not destination.exists():
        fig.savefig(destination)
    plt.close(fig)

    n_bodies = max(
        np_compat
        .asarray(history["final_parameters"]["cos_coefs"])
        .reshape(-1, history["n_modes"] + 1)
        .shape[0]
        for history, _ in results
    )
    fig, axes = plt.subplots(
        2 * n_bodies,
        count,
        figsize=(4 * count, 2.5 * 2 * n_bodies),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for column, alpha in enumerate(alphas):
        for history, _ in results:
            if history["alpha_reg"] != alpha:
                continue
            cos = np_compat.asarray(history["final_parameters"]["cos_coefs"])
            sin = np_compat.asarray(history["final_parameters"]["sin_coefs"])
            if cos.ndim == 1:
                cos, sin = cos[None, :], sin[None, :]
            frequencies = range(1, sin.shape[1] + 1)
            for body in range(cos.shape[0]):
                cos_magnitude = np_compat.abs(cos[body, 1:])
                sin_magnitude = np_compat.abs(sin[body])
                axes[2 * body, column].semilogy(
                    frequencies,
                    np_compat.where(cos_magnitude > 0, cos_magnitude, np_compat.nan),
                    "o-",
                    color=colors[history["n_modes"]],
                )
                axes[2 * body + 1, column].semilogy(
                    frequencies,
                    np_compat.where(sin_magnitude > 0, sin_magnitude, np_compat.nan),
                    "o-",
                    color=colors[history["n_modes"]],
                )
        axes[0, column].set_title(f"alpha={alpha:g}")
        axes[-1, column].set_xlabel("Mode number")
    for body in range(n_bodies):
        axes[2 * body, 0].set_ylabel(f"Body {body + 1} |cos|")
        axes[2 * body + 1, 0].set_ylabel(f"Body {body + 1} |sin|")
    fig.suptitle("Optimized Fourier coefficient magnitudes")
    fig.legend(handles=handles, loc="lower center", ncols=len(handles))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    destination = path / "optimized_coefficients.svg"
    if not destination.exists():
        fig.savefig(destination)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, count, figsize=(4 * count, 3.5), sharex=True, sharey=True, squeeze=False
    )
    for column, alpha in enumerate(alphas):
        ax = axes[0, column]
        for history, _ in results:
            if history["alpha_reg"] == alpha:
                ax.semilogy(history["val_hist"], color=colors[history["n_modes"]])
        ax.set(xlabel="Iteration", title=f"alpha={alpha:g}")
    axes[0, 0].set_ylabel("Objective value")
    fig.suptitle("Optimization history")
    fig.legend(handles=handles, loc="lower center", ncols=len(handles))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    destination = path / "optimization_history.svg"
    if not destination.exists():
        fig.savefig(destination)
    plt.close(fig)
