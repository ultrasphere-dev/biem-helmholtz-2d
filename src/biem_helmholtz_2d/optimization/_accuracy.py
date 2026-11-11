"""Convergence study for point-field and shape-sensitivity quadratures."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import orjson
from array_api.latest import Array
from array_api_compat import array_namespace
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

from biem_helmholtz_2d import accuracy
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
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
    solve_orders: Sequence[int],
    quadrature_orders: Sequence[int],
    reference_order: int,
    step: float,
    n_modes: int,
    seed: int,
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
        values = xp.asarray(
            [[[rng.uniform(-1, 1) for _ in range(2)] for _ in range(n_modes)] for _ in centers],
            dtype=dtype,
            device=device,
        )
        return values / xp.sum(xp.linalg.vector_norm(values, axis=-1), axis=-1)[:, None, None]

    def shapes(coefficients: Array, *, perturbation: bool) -> list[ParameterShape]:
        return [
            ParameterShape(
                center=xp.zeros_like(center) if perturbation else center,
                cos_coefs=xp.concat([
                    xp.full((1,), 0 if perturbation else 1, dtype=dtype, device=device),
                    coefficients[i, :, 0],
                ]),
                sin_coefs=coefficients[i, :, 1],
            )
            for i, center in enumerate(centers)
        ]

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
    for case, perturbation in (("highest_mode", highest), ("random", random_coefficients())):
        data = accuracy(
            shape=shapes(coefficients, perturbation=False),
            h_shape=shapes(perturbation, perturbation=True),
            point=point,
            target=target,
            k=wave_number,
            alpha=coupling_alpha,
            eta=coupling_eta,
            incident_field=incident,
            incident_field_grad=incident_grad,
            solve_orders=solve_orders,
            quadrature_orders=quadrature_orders,
            reference_order=reference_order,
            step=step,
        )
        data["settings"].update({
            "centers": [[float(value) for value in row] for row in centers],
            "coefficients": [
                [[float(value) for value in pair] for pair in body] for body in coefficients
            ],
            "perturbation_coefficients": [
                [[float(value) for value in pair] for pair in body] for body in perturbation
            ],
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


def example_accuracy_plot(path: Path, /) -> None:
    """Plot absolute errors from the ``accuracy.json`` study in ``path``."""
    data = json.loads((path / "accuracy.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    metrics = (
        ("field_error", "Scattered field"),
        ("field_derivative_error", "Field shape derivative (central)"),
        ("objective_derivative_error", "Objective shape derivative (adjoint)"),
    )
    for axis, (key, title) in zip(axes, metrics, strict=True):
        for n_solve in data["settings"]["solve_orders"]:
            records = sorted(
                (row for row in data["records"] if row["n_solve"] == n_solve),
                key=lambda row: row["n_quadrature"],
            )
            axis.loglog(
                [row["n_quadrature"] for row in records],
                [max(row[key], 1e-300) for row in records],
                "o-",
                label=f"solve n={n_solve}",
            )
        orders = sorted(set(data["settings"]["quadrature_orders"]))
        axis.set_xticks(orders, [str(order) for order in orders])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Integration order n (2n-1 nodes)")
        axis.set_ylabel("Absolute error vs. numerical reference")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path / "accuracy.svg")
    plt.close(fig)
