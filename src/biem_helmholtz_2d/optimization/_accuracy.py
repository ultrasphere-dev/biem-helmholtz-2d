"""Convergence study for point-field and shape-sensitivity quadratures."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

import array_api_compat.numpy as np_compat
import orjson
from array_api.latest import Array
from array_api_compat import array_namespace
from ie_circle import trapezoidal_quadrature
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

from biem_helmholtz_2d import accuracy
from biem_helmholtz_2d._acoustic import near_field, scattering_dirichlet
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import field_objective_derivative
from biem_helmholtz_2d._potential_inner import kernel as near_kernel
from biem_helmholtz_2d._potential_inner_derivative import kernel as near_derivative_kernel
from biem_helmholtz_2d.optimization._shape import ParameterShape


def example_accuracy(
    *,
    path: Path,
    centers: Array,
    k: float,
    alpha: float,
    eta: float,
    target_point: tuple[float, float],
    desired_total_field: complex,
    n_solve: Sequence[int],
    n_quad: Sequence[int],
    step: float,
    n_modes: int,
    seed: int,
    n_quad_ref: int | None = None,
) -> None:
    """
    Study random feasible shapes with highest-mode and random perturbations.

    Each body's Fourier coefficient pairs are scaled to the constraint boundary.
    Both cases use the same shapes and have unit perturbation coefficient norm
    per body. The random samples are reproducible from ``seed``; they are stress
    cases, not a proof of the global worst case. Each case is saved in its own
    directory with its coefficients and numerical reference.
    """
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("centers must have shape (n_scatterers, 2) with n_scatterers > 0")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    if step <= 0 or not math.isfinite(step):
        raise ValueError("step must be positive and finite")
    xp = array_namespace(centers)
    dtype, device = centers.dtype, centers.device
    point = xp.asarray(target_point, dtype=dtype, device=device)
    wave_number = xp.asarray(k, dtype=dtype, device=device)
    coupling_alpha = xp.asarray(alpha, dtype=dtype, device=device)
    coupling_eta = xp.asarray(eta, dtype=dtype, device=device)
    direction = xp.asarray([1, 0], dtype=dtype, device=device)
    incident = plane_wave(wave_number, direction)
    incident_grad = plane_wave_grad(wave_number, direction)
    target = desired_total_field - incident(point)
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]  # Reproducible numerical samples, not security.

    def random_coefficients() -> Array:
        values = xp.reshape(
            xp.asarray(
                [rng.uniform(-1, 1) for _ in range(centers.shape[0] * n_modes * 2)],
                dtype=dtype,
                device=device,
            ),
            (centers.shape[0], n_modes, 2),
        )
        return values / xp.sum(xp.linalg.vector_norm(values, axis=-1), axis=-1)[:, None, None]

    def shapes(coefficients: Array, *, perturbation: bool) -> ParameterShape:
        constant = xp.full(
            (centers.shape[0], 1), 0 if perturbation else 1, dtype=dtype, device=device
        )
        return ParameterShape(
            center=xp.zeros_like(centers) if perturbation else centers,
            cos_coefs=xp.concat([constant, coefficients[..., 0]], axis=-1),
            sin_coefs=coefficients[..., 1],
        )

    coefficients = random_coefficients()
    highest = xp.broadcast_to(
        xp.stack(
            [
                xp.concat([
                    xp.zeros(n_modes - 1, dtype=dtype, device=device),
                    xp.ones(1, dtype=dtype, device=device),
                ]),
                xp.zeros(n_modes, dtype=dtype, device=device),
            ],
            axis=-1,
        ),
        coefficients.shape,
    )

    def study(case: str, perturbation: Array, /) -> None:
        shape = shapes(coefficients, perturbation=False)
        h_shape = shapes(perturbation, perturbation=True)

        def evaluate(solve: int, quad: int, /) -> dict[str, Array]:
            def field_for(boundary: ParameterShape, /) -> Array:
                density = scattering_dirichlet(
                    k=wave_number,
                    shape=boundary,
                    incident_field=incident,
                    alpha=coupling_alpha,
                    eta=coupling_eta,
                    n=solve,
                )
                return near_field(
                    density,
                    point,
                    shape=boundary,
                    k=wave_number,
                    alpha=coupling_alpha,
                    eta=coupling_eta,
                    n=quad,
                )

            density = scattering_dirichlet(
                k=wave_number,
                shape=shape,
                incident_field=incident,
                alpha=coupling_alpha,
                eta=coupling_eta,
                n=solve,
            )
            field = near_field(
                density,
                point,
                shape=shape,
                k=wave_number,
                alpha=coupling_alpha,
                eta=coupling_eta,
                n=quad,
            )
            t, _ = trapezoidal_quadrature(
                quad,
                xp=xp,
                device=device,
                dtype=xp.result_type(
                    xp.real(wave_number), xp.real(coupling_alpha), xp.real(coupling_eta)
                ),
            )
            _, directional = field_objective_derivative(
                density,
                point[None, :],
                (2 * (field - target))[None],
                shape=shape,
                incident_gradient=incident_grad(shape.x(t)),
                k=wave_number,
                alpha=coupling_alpha,
                eta=coupling_eta,
                n_solve=solve,
                n_quad=quad,
                kernel=near_kernel,
                derivative_kernel=near_derivative_kernel,
            )
            derivative = directional(h_shape)
            field_plus = field_for(shapes(coefficients + step * perturbation, perturbation=False))
            field_minus = field_for(shapes(coefficients - step * perturbation, perturbation=False))
            return {
                "field": field,
                "field_derivative": (field_plus - field_minus) / (2 * step),
                "objective_derivative": derivative,
                "objective_central": (
                    xp.abs(field_plus - target) ** 2 - xp.abs(field_minus - target) ** 2
                )
                / (2 * step),
            }

        data = accuracy(evaluate, n_solve=n_solve, n_quad=n_quad, n_quad_ref=n_quad_ref)
        data["settings"]["step"] = step
        data["settings"].update({
            "centers": np_compat.asarray(xp.asarray(centers, device="cpu")).tolist(),
            "coefficients": np_compat.asarray(xp.asarray(coefficients, device="cpu")).tolist(),
            "perturbation_coefficients": np_compat.asarray(
                xp.asarray(perturbation, device="cpu")
            ).tolist(),
            "case": case,
            "seed": seed,
            "k": k,
            "alpha": alpha,
            "eta": eta,
            "target_point": target_point,
            "desired_total_field": {
                "real": desired_total_field.real,
                "imag": desired_total_field.imag,
            },
        })
        directory = path / case
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "accuracy.json").write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    study("highest_mode", highest)
    study("random", random_coefficients())


def example_accuracy_plot(path: Path, /) -> None:
    """Plot absolute errors from the ``accuracy.json`` study in ``path``."""
    data = json.loads((path / "accuracy.json").read_text())
    for row in data["records"]:
        if "n_quad" not in row:
            row["n_quad"] = row["n_quadrature"]
    settings = data["settings"]
    n_solve = settings["n_solve"] if "n_solve" in settings else settings["solve_orders"]
    n_quad = settings["n_quad"] if "n_quad" in settings else settings["quadrature_orders"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    metrics = (
        ("field_error", "Scattered field"),
        ("field_derivative_error", "Field shape derivative (central)"),
        ("objective_derivative_error", "Objective shape derivative (adjoint)"),
    )
    for axis, (key, title) in zip(axes, metrics, strict=True):
        for solve in n_solve:
            records = sorted(
                (row for row in data["records"] if row["n_solve"] == solve),
                key=lambda row: row["n_quad"],
            )
            axis.loglog(
                [row["n_quad"] for row in records],
                [max(row[key], 1e-300) for row in records],
                "o-",
                label=f"solve n={solve}",
            )
        values = sorted(set(n_quad))
        axis.set_xticks(values, [str(value) for value in values])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Integration n (2n-1 nodes)")
        axis.set_ylabel("Absolute error vs. numerical reference")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path / "accuracy.svg")
    plt.close(fig)
